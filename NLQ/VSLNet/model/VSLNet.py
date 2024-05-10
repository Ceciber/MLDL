"""VSLNet Baseline for Ego4D Episodic Memory -- Natural Language Queries.
"""
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from model.layers import (
    # Imports a class or function related to word or token embeddings
    Embedding,
    # Imports something that might involve projecting visual input data into a format compatible with the neural network
    VisualProjection,
    # Feature encoder
    FeatureEncoder,
    # Imports a class/method related to Context query Attention
    CQAttention,
    # Concatenate context and query information
    CQConcatenate,
    # 
    ConditionedPredictor,
    # HighLightLayer,
    BertEmbedding,
)


def build_optimizer_and_scheduler(model, configs):
    # specify a list of parameters for which we don't want to apply weight decays because it's not desirable
    no_decay = [
        "bias",
        "layer_norm",
        "LayerNorm",
    ]  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                # for any other parameters not in the "no_decay" list we apply the weight decay
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    # After organizing the parameters, it initializes the optimizer.
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    #  This scheduler adjusts the learning rate during training, starting with a warm-up period where the learning rate increases linearly, followed by a linear decay
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        configs.num_train_steps * configs.warmup_proportion,
        configs.num_train_steps,
    )
    return optimizer, scheduler


class VSLNet(nn.Module):
    # The constructor initializes the parameters of the VSLNet class. It takes two arguments: configs, which likely contains configuration parameters 
    # for the model, and word_vectors, which may contain pre-trained word embeddings.
    def __init__(self, configs, word_vectors):
        super(VSLNet, self).__init__()
        self.configs = configs

        # This initializes a VisualProjection layer, which likely projects visual features into a lower-dimensional space.
        self.video_affine = VisualProjection(
            visual_dim=configs.video_feature_dim,
            dim=configs.dim,
            drop_rate=configs.drop_rate,
        )

        # This initializes a FeatureEncoder layer, which encodes features, possibly both textual and visual, using self-attention mechanisms.
        # Paper: the encoder consists of four convolutional layers, followed by a multi-head attention layer, and a feed forward layer; normalization applied to each layer
        self.feature_encoder = FeatureEncoder(
            dim=configs.dim,
            num_heads=configs.num_heads,
            kernel_size=7,
            num_layers=4,
            max_pos_len=configs.max_pos_len,
            drop_rate=configs.drop_rate,
        )
        # video and query fusion
        self.cq_attention = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concat = CQConcatenate(dim=configs.dim)
        # query-guided highlighting
        """ self.highlight_layer = HighLightLayer(dim=configs.dim) """
        # conditioned predictor
        self.predictor = ConditionedPredictor(
            dim=configs.dim,
            num_heads=configs.num_heads,
            drop_rate=configs.drop_rate,
            max_pos_len=configs.max_pos_len,
            predictor=configs.predictor,
        )

        # If pretrained transformer, initialize_parameters and load.
        if configs.predictor == "bert":
            # Project back from BERT to dim.
            self.query_affine = nn.Linear(768, configs.dim)
            # init parameters
            self.init_parameters()
            self.embedding_net = BertEmbedding(configs.text_agnostic)
        else:
            self.embedding_net = Embedding(
                num_words=configs.word_size,
                num_chars=configs.char_size,
                out_dim=configs.dim,
                word_dim=configs.word_dim,
                char_dim=configs.char_dim,
                word_vectors=word_vectors,
                drop_rate=configs.drop_rate,
            )
            # init parameters
            self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Linear)
            ):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()

        self.apply(init_weights)

    
    # Takes as argument the input data
    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        video_features = self.video_affine(video_features)
        # It specifies how the input data should be processed through the layers of the model to produce the desired output.
        if self.configs.predictor == "bert":
            query_features = self.embedding_net(word_ids)
            query_features = self.query_affine(query_features)
        else:
            query_features = self.embedding_net(word_ids, char_ids)

        query_features = self.feature_encoder(query_features, mask=q_mask)
        video_features = self.feature_encoder(video_features, mask=v_mask)
        features = self.cq_attention(video_features, query_features, v_mask, q_mask)
        features = self.cq_concat(features, query_features, q_mask)
        """ h_score = self.highlight_layer(features, v_mask) 
        features = features * h_score.unsqueeze(2) """
        start_logits, end_logits = self.predictor(features, mask=v_mask)
        # return h_score, start_logits, end_logits
        return start_logits, end_logits
        # "logits" typically refer to the raw, unnormalized prediction scores produced by a model before applying a softmax function

    # once we have the raw prediction scores, we pass them to the predictor that contains logic to extract the indices corresponding to start and end.
    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(
            start_logits=start_logits, end_logits=end_logits
        )

    """ def compute_highlight_loss(self, scores, labels, mask):
            return self.highlight_layer.compute_loss(
                scores=scores, labels=labels, mask=mask
        ) """

    # The method delegates the computation of the prediction loss to the predictor component of the model, which contains logic for computing the cross-entropy loss based on the predicted logits and the target labels.
    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            start_labels=start_labels,
            end_labels=end_labels,
        )
