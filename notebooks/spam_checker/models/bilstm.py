import torch
import torch.nn as nn

class BILSTMTextSpamClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1, n_layers=2, dropout=0.5):
        super(BILSTMTextSpamClassifier, self).__init__()
        
        # 1. Embedding Layer: Converts word indices to dense vectors
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        # 2. BiLSTM Layer: Processes sequences bidirectionally
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # 3. Fully Connected Layer: Maps LSTM outputs to classification probabilities
        # Multiplied by 2 because BiLSTM has both forward and backward hidden states
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # 4. Sigmoid Activation: Squashes the output to a 0-1 range (probability)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text, text_lengths):
        # text shape: [batch_size, sequence_length]
        
        # Pass tokens through embedding layer
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        
        # Pack padded sequences to tell LSTM to ignore padding tokens
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack the sequence
        # output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Concatenate the final forward and backward hidden layers
        cat_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # Apply dropout
        dropped = self.dropout(cat_hidden)
        
        # Pass through classifier
        dense_outputs = self.fc(dropped)
        
        # Apply sigmoid for binary classification
        outputs = self.sigmoid(dense_outputs)
        
        return outputs
