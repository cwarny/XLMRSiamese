def mean_encoded_seq_batch(encoded, seq, ignore_index=1):
    '''
    Computes mean of encoded seq while ignoring padding
    encoded (batch_size,seq_len,emb_dim)
    seq (batch_size,seq_len)
    '''
    mask = seq!=ignore_index
    masked = (encoded*mask.unsqueeze(-1))
    return masked.sum(1)/mask.sum(1).unsqueeze(-1)