from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import Constant
from networks.layers import (AdaIN, AdaptiveAttention, AdaptiveAttentionSOA,
                             InstanceNormalization)

import numpy as np


def residual_down_block(inputs, filters, resample=True):
    x = inputs # crea una copia di inputs perche da questo punto nasceranno due percorsi diversi, uno per la convoluzione e uno per il residuo

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x) # creazione della connessione residua, modificando il numero di canali con un kernel 1x1, 
                                                                             # stride 1 e padding 'same' per mantenere le dimensioni spaziali
    if resample:
        r = AveragePooling2D()(r) #DownSampling riducendo le dimensioni spaziali di r di un fattore 2

    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = AveragePooling2D()(x)

    x = Add()([x, r]) # --> connesiione residua: somma il percorso convoluzionale con il percorso residuo

    return x


def residual_up_block(inputs, filters, resample=True, name=None):
    x, z_id = inputs

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = UpSampling2D(interpolation='bilinear')(r)

    x = InstanceNormalization()(x)
    x = AdaIN()([x, z_id])
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = UpSampling2D(interpolation='bilinear')(x)

    x = Add()([x, r])

    return x


def adaptive_attention(inputs, filters, name=None):
    x_t, x_s = inputs

    m = Concatenate(axis=-1)([x_t, x_s])
    m = Conv2D(filters=filters // 4, kernel_size=3, strides=1, padding='same')(m)
    m = LeakyReLU(0.2)(m)
    m = InstanceNormalization()(m)
    m = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', activation='sigmoid', name=name)(m)

    x = AdaptiveAttention()([m, x_t, x_s])

    return x


def adaptive_attention_double(inputs, filters, name=None):
    x_t, x_s = inputs

    c = Concatenate(axis=-1)([x_t, x_s])

    m_hat = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(c)
    m_hat = LeakyReLU(0.2)(m_hat)
    m_hat = InstanceNormalization()(m_hat)
    m_hat = Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation='sigmoid', name=name + '_hat')(m_hat)

    m = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(c)
    m = LeakyReLU(0.2)(m)
    m = InstanceNormalization()(m)
    m = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', activation='sigmoid', name=name)(m)

    m_hat = m * m_hat

    x = AdaptiveAttention()([m_hat, x_t, x_s])

    return x


def adaptive_attention_soa(inputs, name=None):
    """Fuse target attributes, identity-conditioned and occluder features."""
    x_attr, x_id, x_occ = inputs

    c = Concatenate(axis=-1)([x_attr, x_id, x_occ])
    hidden_filters = max(int(x_id.shape[-1]) // 4, 16)
    gates = Conv2D(filters=hidden_filters, kernel_size=3, strides=1,
                   padding='same')(c)
    gates = LeakyReLU(0.2)(gates)
    gates = InstanceNormalization()(gates)
    gates = Conv2D(filters=3, kernel_size=1, strides=1, padding='same',
                   activation='softmax',
                   bias_initializer=Constant([0.0, 0.0, -4.0]),
                   name=name)(gates)

    return AdaptiveAttentionSOA()([gates, x_attr, x_id, x_occ])


def adaptive_fusion_up_block(inputs, filters, resample=True, name=None):
    x_t, x_s, z_id = inputs

    x = adaptive_attention([x_t, x_s], x_t.shape[-1], name=name)

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = UpSampling2D(interpolation='bilinear')(r)

    x = InstanceNormalization()(x)
    x = AdaIN()([x, z_id])
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = UpSampling2D(interpolation='bilinear')(x)

    x = Add()([x, r])

    return x


def adaptive_fusion_up_block_double(inputs, filters, resample=True, name=None):
    x_t, x_s, z_id = inputs

    x = adaptive_attention_double([x_t, x_s], x_t.shape[-1], name=name)

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = UpSampling2D(interpolation='bilinear')(r)

    x = InstanceNormalization()(x)
    x = AdaIN()([x, z_id])
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = UpSampling2D(interpolation='bilinear')(x)

    x = Add()([x, r])

    return x


def adaptive_fusion_up_block_soa(inputs, filters, resample=True, name=None):
    x_attr, x_id, x_occ, z_id = inputs

    attention_name = None if name is None else name + '_soa'
    x = adaptive_attention_soa([x_attr, x_id, x_occ], name=attention_name)

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = UpSampling2D(interpolation='bilinear')(r)

    x = InstanceNormalization()(x)
    x = AdaIN()([x, z_id])
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = UpSampling2D(interpolation='bilinear')(x)

    return Add()([x, r])


def dual_adaptive_fusion_up_block(inputs, filters, resample=True, name=None):
    x_t, x_s, z_id = inputs

    x = adaptive_attention([x_t, x_s], x_t.shape[-1], name=name + '_0')
    x = adaptive_attention([x_t, x], x_t.shape[-1], name=name + '_1')

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = UpSampling2D(interpolation='bilinear')(r)

    x = InstanceNormalization()(x)
    x = AdaIN()([x, z_id])
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = UpSampling2D(interpolation='bilinear')(x)

    x = Add()([x, r])

    return x


def adaptive_fusion_up_block_concat_baseline(inputs, filters, resample=True, name=None):
    x_t, x_s, z_id = inputs

    x = Concatenate(axis=-1)([x_t, x_s])

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = UpSampling2D(interpolation='bilinear')(r)

    x = InstanceNormalization()(x)
    x = AdaIN()([x, z_id])
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = UpSampling2D(interpolation='bilinear')(x)

    x = Add(name=name if name == 'final' else None)([x, r])

    return x


def adaptive_fusion_up_block_add_baseline(inputs, filters, resample=True, name=None):
    x_t, x_s, z_id = inputs

    x = Add()([x_t, x_s])

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = UpSampling2D(interpolation='bilinear')(r)

    x = InstanceNormalization()(x)
    x = AdaIN()([x, z_id])
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = UpSampling2D(interpolation='bilinear')(x)

    x = Add()([x, r])

    return x


# helper function for choosing feature fusion method
def make_layer(l_type, inputs, filters, resample, name=None):
    if l_type == 'affa_soa':
        return adaptive_fusion_up_block_soa(inputs, filters, resample=resample, name=name)
    if l_type == 'affa':
        return adaptive_fusion_up_block(inputs, filters, resample=resample, name=name)
    if l_type == 'd_affa':
        return dual_adaptive_fusion_up_block(inputs, filters, resample=resample, name=name)
    if l_type == 'do_affa':
        return adaptive_fusion_up_block_double(inputs, filters, resample=resample, name=name)
    elif l_type == 'concat':
        return adaptive_fusion_up_block_concat_baseline(inputs, filters, resample=resample, name=name)
    elif l_type == 'add':
        return adaptive_fusion_up_block_add_baseline(inputs, filters, resample=resample, name=name)
    elif l_type == 'no_skip':
        return residual_up_block(inputs[1:], filters, resample=resample)

def get_occlusion_encoder():
    """
    Input:
        C: [B, 256, 256, 4]

    Output in ordine decoder:
        8, 16, 32, 64, 128, 256.
    """
    condition = Input(
        shape=(256, 256, 4),
        name='occ_encoder_input'
    )

    f_256 = Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        name='occ_256_conv'
    )(condition)
    f_256 = LeakyReLU(0.2, name='occ_256_act')(f_256)

    f_128 = residual_down_block(f_256, 128)
    f_64 = residual_down_block(f_128, 256)
    f_32 = residual_down_block(f_64, 512)
    f_16 = residual_down_block(f_32, 512)
    f_8 = residual_down_block(f_16, 512)

    return Model(
        inputs=condition,
        outputs=[
            f_8,
            f_16,
            f_32,
            f_64,
            f_128,
            f_256
        ],
        name='E_occ'
    )


def get_generator(up_types=None, mapping_depth=4, mapping_size=256):

    # if up_types=None, use a default setting
    if up_types is None:
        up_types = ['affa_soa', 'affa_soa', 'affa_soa',
                    'affa_soa', 'affa_soa', 'affa_soa'] #['no_skip', 'no_skip', 'affa', 'affa', 'affa', 'concat']

    x_target = Input(shape=(256, 256, 3))
    z_source = Input(shape=(512,))

    c_source = Input(shape=(256, 256, 4), name='source_occlusion_condition')

    E_occ = get_occlusion_encoder()
    occ_features = E_occ(c_source)

    occ_8, occ_16, occ_32, occ_64, occ_128, occ_256 = occ_features

    # build mapping network M
    z_id = z_source
    for m in range(np.max([mapping_depth - 1, 0])):
        z_id = Dense(mapping_size)(z_id)
        z_id = LeakyReLU(0.2)(z_id)
    if mapping_depth >= 1:
        z_id = Dense(mapping_size)(z_id)

    # build generator network G
    x_0 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x_target)            # 256

    x_1 = residual_down_block(x_0, 128)                                                     # 128

    x_2 = residual_down_block(x_1, 256)                                                     # 64

    x_3 = residual_down_block(x_2, 512)                                                     # 32

    x_4 = residual_down_block(x_3, 512)                                                     # 16

    x_5 = residual_down_block(x_4, 512)                                                     # 8

    x_6 = residual_down_block(x_5, 512, resample=False)                                     # 8

    u_5 = residual_up_block([x_6, z_id], 512, resample=False)                               # 8

    def decoder_layer(layer_type, x_attr, x_id, x_occ, filters, resample, name):
        if layer_type == 'affa_soa':
            inputs = [x_attr, x_id, x_occ, z_id]
        else:
            inputs = [x_attr, x_id, z_id]
            
        return make_layer(layer_type, inputs, filters, resample=resample, name=name)

    u_4 = decoder_layer(up_types[0], x_5, u_5, occ_8,
                        512, True, '16x16')                                                # 16

    u_3 = decoder_layer(up_types[1], x_4, u_4, occ_16,
                        512, True, '32x32')                                                # 32

    u_2 = decoder_layer(up_types[2], x_3, u_3, occ_32,
                        256, True, '64x64')                                                # 64

    u_1 = decoder_layer(up_types[3], x_2, u_2, occ_64,
                        128, True, '128x128')                                              # 128

    u_0 = decoder_layer(up_types[4], x_1, u_1, occ_128,
                        64, True, '256x256')                                               # 256

    out = decoder_layer(up_types[5], x_0, u_0, occ_256,
                        3, False, 'final')                                                 # 256

    gen_model = Model([x_target, z_source, c_source], out, name='FaceDancer_SOA')
    gen_model.summary()

    return gen_model





