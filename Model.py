import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

USE_CUDA = False

class CNN(torch.nn.Module):
	def __init__(self, nwords, emb_size, num_filters, window_sizes, dropout, ntags, weight_norm, Type, pretrained_embedding=None):
		super(CNN, self).__init__()

		self.name = 'CNN'
		""" layers """
		self.embedding = torch.nn.Embedding(nwords, emb_size)
		if pretrained_embedding is not None:
			self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding).type(Type))
		else:
			# uniform initialization
			torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)

		# Conv 1d
		self.conv_1d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_sizes[0],
									   stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv_2d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_sizes[1],
									   stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv_3d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_sizes[2],
									   stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.relu = torch.nn.ReLU()
		
		# Drop out layer
		self.drop_layer = torch.nn.Dropout(p=dropout)
		self.projection_layer = torch.nn.Linear(in_features=3*num_filters, out_features=ntags, bias=True)
		# self.projection_layer = torch.nn.Linear(in_features=num_filters, out_features=ntags, bias=True)
		# Initializing the projection layer
		torch.nn.init.xavier_uniform_(self.projection_layer.weight)
		self.weight_norm = weight_norm
		self.embedding.weight.requires_grad = True
		self.min_len = max(window_sizes)

	def forward(self, words, lengths=0, return_activations=False, padded=0):
    	# add argument lengths to consist with RNN model
		if words.size()[1] < self.min_len:
			words = F.pad(input=words, pad=(0, self.min_len - words.size()[1], 0, 0), mode='constant', value=padded)
		emb = self.embedding(words)                 # nwords x emb_size
		if len(emb.size()) == 3:
			batch = emb.size()[0]
			emb = emb.permute(0, 2, 1)
		else:
			batch = 1
			emb = emb.unsqueeze(0).permute(0, 2, 1)     # 1 x emb_size x nwords

		# emb of size [batch, embedding_size, sentence_length]
		# h of size [batch, filter_size, sentence_length - window_size + 1]
		h1 = self.conv_1d(emb).max(dim=2)[0]
		h2 = self.conv_2d(emb).max(dim=2)[0]
		h3 = self.conv_3d(emb).max(dim=2)[0]

		# h_flat = h1
		h_flat = torch.cat([h1, h2, h3], dim=1)                    # [batch, 3*filter]

		# activation operation receives size of [batch, filter_size, sentence_length - window_size + 1]
		# activation [batch, sentence_length - window_size + 1] argmax along length of the sentence
		# the max operation reduce the filter_size dimension and select the index ones
		activations = h_flat.max(dim=1)[1]

		# Do max pooling
		h_flat = self.relu(h_flat)
		features = h_flat.squeeze(0)               # [batch, 3*filter]
		h = self.drop_layer(features)
		out = self.projection_layer(h)              # size(out) = 1 x ntags
		if return_activations:
			return out, activations.data.cpu().numpy(), features.data.cpu().numpy()
		return out



class Attn(nn.Module):
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()

		self.method = method
		self.hidden_size = hidden_size

		if self.method == 'general':
			self.attn = nn.Linear(self.hidden_size, hidden_size)

		elif self.method == 'concat':
			self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
			self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

	def forward(self, hidden, encoder_outputs):
		max_len = encoder_outputs.size(0)
		this_batch_size = encoder_outputs.size(1)

		# Create variable to store attention energies
		attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S

		if USE_CUDA:
			attn_energies = attn_energies.cuda()

		# For each batch of encoder outputs
		for b in range(this_batch_size):
			# Calculate energy for each encoder output
			for i in range(max_len):
				attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

		# Normalize energies to weights in range 0 to 1, resize to 1 x B x S
		return F.softmax(attn_energies).unsqueeze(1)

	# slower than GPU
	def score(self, hidden, encoder_output):
		if self.method == 'dot':
			energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
		elif self.method == 'general':
			energy = self.attn(encoder_output)
			energy = torch.dot(hidden.view(-1), energy.view(-1))
		elif self.method == 'concat':
			energy = self.attn(torch.cat((hidden, encoder_output), 1))
			energy = torch.dot(self.v.view(-1), energy.view(-1))
		return energy

class LuongAttnDecoderRNN(nn.Module):
	def __init__(self, attn_model, hidden_size, output_size_set, external_embedding=None, n_layers=1, dropout=0.1):
		super(LuongAttnDecoderRNN, self).__init__()

		# Keep for reference
		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.output_size = output_size_set
		self.n_layers = n_layers
		self.dropout = dropout

		# Define layers
		if external_embedding is not None:
			self.embedding = external_embedding
		else:
			self.embedding = nn.Embedding(output_size_set['pinyin'], hidden_size)
		#             self.embedding = {}
		#             for category, v in output_size_set.items():
		#                 self.embedding[category] = nn.Embedding(v, hidden_size)
		self.embedding_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = {}
		for category, output_size in output_size_set.items():
			self.out[category] = nn.Linear(hidden_size, output_size)

		# Choose attention model
		if attn_model != 'none':
			self.attn = Attn(attn_model, hidden_size)

	def forward(self, input_seq, last_hidden, encoder_outputs):
		# Note: we run this one step at a time

		# Get the embedding of the current input word (last output word)
		batch_size = input_seq.size(0)
		embedded = self.embedding(input_seq)
		embedded = self.embedding_dropout(embedded)
		embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N

		# Get current hidden state from input word and last hidden state
		try:
			rnn_output, hidden = self.gru(embedded, last_hidden)
		except:
			pdb.set_trace()

		# Calculate attention from current RNN state and all encoder outputs;
		# apply to encoder outputs to get weighted average
		attn_weights = self.attn(rnn_output, encoder_outputs)
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

		# Attentional vector using the RNN hidden state and context vector
		# concatenated together (Luong eq. 5)
		rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
		context = context.squeeze(1)  # B x S=1 x N -> B x N
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = torch.tanh(self.concat(concat_input))

		# Finally predict next token (Luong eq. 6, without softmax)
		output = {}
		for category in self.out.keys():
			output[category] = self.out[category](concat_output)

		# Return final output, hidden state, and attention weights (for visualization)
		return output, hidden, attn_weights

def sequence_mask(sequence_length, max_len=None):
	if max_len is None:
		max_len = sequence_length.data.max()
	batch_size = sequence_length.size(0)
	# seq_range = torch.range(0, max_len - 1).long()
	seq_range = torch.arange(0, max_len).long()
	seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
	seq_range_expand = Variable(seq_range_expand)
	if sequence_length.is_cuda:
		seq_range_expand = seq_range_expand.cuda()
	seq_length_expand = (sequence_length.unsqueeze(1)
						 .expand_as(seq_range_expand))

	return seq_range_expand < seq_length_expand

def Masked_cross_entropy(logits, target, length):
	# e.g. logits of (20, 6, 45), target of (20, 6), length of size 20 of max 6
	length = Variable(torch.LongTensor(length))
	# logits_flat: (batch * max_len, num_classes)
	logits_flat = logits.view(-1, logits.size(-1))
	# log_probs_flat: (batch * max_len, num_classes)
	log_probs_flat = F.log_softmax(logits_flat)
	# target_flat: (batch * max_len, 1)
	target_flat = target.view(-1, 1)
	# losses_flat: (batch * max_len, 1)
	losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
	# losses: (batch, max_len)
	losses = losses_flat.view(*target.size())
	# mask: (batch, max_len)
	mask = sequence_mask(sequence_length=length, max_len=target.size(1))
	losses = torch.mul(losses, mask.float())
	loss = torch.div(torch.sum(losses), torch.sum(length.float()))
	return loss