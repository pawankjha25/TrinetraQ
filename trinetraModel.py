import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input


# Self Attention layer
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % self.num_heads == 0
        self.projection_dim = embed_size // num_heads
        self.query_dense = layers.Dense(embed_size)
        self.key_dense = layers.Dense(embed_size)
        self.value_dense = layers.Dense(embed_size)
        self.combine_heads = layers.Dense(embed_size)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_size))
        output = self.combine_heads(concat_attention)
        return output

# Building the Transformer block
class TrinetraTransformer(layers.Layer):
    def __init__(self, embed_size, num_heads, ff_dim, rate=0.1,  **kwargs):
        """
                Args:
                    embed_size (int): Size of the embedding.
                    num_heads (int): Number of attention heads.
                    ff_dim (int): Dimension of the feed-forward network.
                    rate (float): Dropout rate.
                    **kwargs: Additional arguments for the base Layer class (e.g., trainable, dtype).
                """
        super(TrinetraTransformer, self).__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_size, num_heads)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_size), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.projection = layers.Dense(embed_size)  # New projection layer

    def call(self, inputs, training):
        print(f"Input shape: {inputs.shape}")
        inputs_proj = self.projection(inputs)  # Project input to match embed_size
        attn_output = self.att(inputs_proj)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs_proj + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

