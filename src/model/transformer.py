# src/model/transformer.py
import torch
import torch.nn as nn

class SudokuTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # [수정] 함수 인자에서 기본값(d_model=512, num_layers=12 등)을 삭제했습니다.
        # 대신 무조건 config 객체에서 값을 가져오도록 통일했습니다.
        # 이제 config.py 하나만 수정하면 모든 게 바뀝니다.
        
        d_model = config.D_MODEL
        nhead = config.NHEAD
        num_layers = config.NUM_LAYERS
        dropout = config.DROPOUT
        activation = config.ACTIVATION
        
        self.seq_len = config.SEQ_LEN
        self.grid_size = config.GRID_SIZE
        self.box_h = config.BOX_H
        self.box_w = config.BOX_W
        
        # 임베딩
        self.token_embedding = nn.Embedding(config.NUM_CLASSES, d_model)
        self.row_embedding = nn.Embedding(self.grid_size, d_model)
        self.col_embedding = nn.Embedding(self.grid_size, d_model)
        self.box_embedding = nn.Embedding(self.grid_size, d_model)
        
        # 구조적 어텐션 마스크
        self.register_buffer('attn_mask', self._generate_sudoku_mask())

        # 인코더 (GELU 등 설정도 Config에서 가져옴)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout, 
            activation=activation,
            batch_first=True, 
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, config.NUM_CLASSES)

        # 인덱스 버퍼
        self.register_buffer('row_idx', torch.arange(self.seq_len) // self.grid_size)
        self.register_buffer('col_idx', torch.arange(self.seq_len) % self.grid_size)
        self.register_buffer('box_idx', 
            (torch.arange(self.seq_len) // (self.grid_size * self.box_h)) * self.box_h + 
            (torch.arange(self.seq_len) % self.grid_size) // self.box_w
        )

    def _generate_sudoku_mask(self):
        mask = torch.full((self.seq_len, self.seq_len), float('-inf'))
        for i in range(self.seq_len):
            r1, c1 = i // self.grid_size, i % self.grid_size
            b1 = (r1 // self.box_h) * self.box_h + (c1 // self.box_w)
            for j in range(self.seq_len):
                r2, c2 = j // self.grid_size, j % self.grid_size
                b2 = (r2 // self.box_h) * self.box_h + (c2 // self.box_w)
                if r1 == r2 or c1 == c2 or b1 == b2:
                    mask[i, j] = 0.0
        return mask

    def forward(self, x):
        if x.dim() == 3: x = x.view(x.size(0), -1)
        
        x = self.token_embedding(x)
        pos = self.row_embedding(self.row_idx) + \
              self.col_embedding(self.col_idx) + \
              self.box_embedding(self.box_idx)
        x = x + pos
        
        x = self.transformer_encoder(x, mask=self.attn_mask)
        logits = self.output_layer(x)
        return logits