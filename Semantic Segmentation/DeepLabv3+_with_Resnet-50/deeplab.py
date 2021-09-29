import tensorflow as tf
#from tensorflow.keras import backend as K
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import AveragePooling2D, Lambda, Conv2D, Conv2DTranspose, Activation, Reshape, concatenate, Concatenate, BatchNormalization, ZeroPadding2D
#from resnet50 import ResNet50
import keras
from keras.models import *
from keras.layers import *
from keras import layers
import keras.backend as K
from resnet.resnet501 import get_resnet50_encoder


def Upsample(tensor, size):
    '''bilinear upsampling'''
    name = tensor.name.split('/')[0] + '_upsample'

    def bilinear_upsample(x, size):
        resized = tf.image.resize(
            images=x, size=size)
        return resized
    y = Lambda(lambda x: bilinear_upsample(x, size),
               output_shape=size, name=name)(tensor)
    return y


def aspp(x):
    shape_before =K.int_shape
    #shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    # upsample. have to use compat because of the option align_corners
    
    size_before = K.int_shape(x)
    #size_before = tf.keras.backend.int_shape(x)
    #b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
    #                                                method='bilinear', align_corners=True))(b4)
    
       
    b4 = Lambda(lambda x: tf.keras.backend.resize_images(x, size_before[1],size_before[2], data_format='channels_last',interpolation ='bilinear'))(b4) 


    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    #b0 = Activation(tf.nn.relu, name='aspp0_activation')(b0)
    b0 = Activation('relu',name='aspp0_activation')(b0)
    # there are only 2 branches in mobilenetV2. not sure why

    # rate = 6 (12)
    b1 = SeparableConv2D(filters=256, kernel_size=1, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', name='aspp1', use_bias=False)(x)
    
    
    
    b2 = SeparableConv2D(filters=256, kernel_size=1, dilation_rate=12, padding='same',
                 kernel_initializer='he_normal', name='aspp2', use_bias=False)(x)
        
        
    b3 = SeparableConv2D(filters=256, kernel_size=1, dilation_rate=18, padding='same',
                 kernel_initializer='he_normal', name='aspp3', use_bias=False)(x)
            


    # concatenate ASPP branches & project
    #x = Concatenate()([b4, b0, b1, b2, b3])
    y = concatenate([b4, b0, b1, b2, b3], name='aspp_concat')
    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(y)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x) 
    x = Dropout(0.1)(x)
    print(x.shape)
    return x



def ASPP(tensor):
    '''atrous spatial pyramid pooling'''
    dims = K.int_shape(tensor)
    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', name='pool_1x1conv2d', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)

    y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d1', use_bias=False)(tensor)
    y_1 = BatchNormalization(name=f'bn_2')(y_1)
    y_1 = Activation('relu', name=f'relu_2')(y_1)

    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d6', use_bias=False)(tensor)
    y_6 = BatchNormalization(name=f'bn_3')(y_6)
    y_6 = Activation('relu', name=f'relu_3')(y_6)

    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d12', use_bias=False)(tensor)
    y_12 = BatchNormalization(name=f'bn_4')(y_12)
    y_12 = Activation('relu', name=f'relu_4')(y_12)

    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d18', use_bias=False)(tensor)
    y_18 = BatchNormalization(name=f'bn_5')(y_18)
    y_18 = Activation('relu', name=f'relu_5')(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', name='ASPP_conv2d_final', use_bias=False)(y)
    y = BatchNormalization(name=f'bn_final')(y)
    y = Activation('relu', name=f'relu_final')(y)
    return y


def DeepLabV3Plus(img_height, img_width, nclasses=66):
    print('*** Building DeepLabv3Plus Network ***')
    height=8         #img_height//48
    width = 8         #  img_width//64
    #base_model = ResNet50(input_shape=(
     #   img_height, img_width, 4), weights='imagenet', include_top=False)

    base_model = get_resnet50_encoder(input_height=img_height,  input_width=img_width)
    #base_model.summary()

    image_features = base_model.get_layer('activation_48').output
    x_a = aspp(image_features)
    #x_a = Upsample(tensor=x_a, size=[img_height // 4, img_width // 4])
    
    x_a = Lambda(lambda x: tf.keras.backend.resize_images(x, height, width, data_format='channels_last',interpolation ='bilinear'))(x_a) 
    
     
   

    x_b = base_model.get_layer('activation_9').output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same',
                 kernel_initializer='he_normal', name='low_level_projection', use_bias=False)(x_b)
    x_b = BatchNormalization(name=f'bn_low_level_projection')(x_b)
    x_b = Activation('relu', name='low_level_activation')(x_b)
   
    x = concatenate([x_a, x_b], name='decoder_concat')
    #x = Concatenate()([x_a, x_b])
     
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_1', use_bias=False)(x)
    x = BatchNormalization(name=f'bn_decoder_1')(x)
    x = Activation('relu', name='activation_decoder_1')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_2', use_bias=False)(x)
    x = BatchNormalization(name=f'bn_decoder_2')(x)
    x_s = Activation('relu', name='activation_decoder_2')(x)
    

    
    x_s = Lambda(lambda x: tf.keras.backend.resize_images(x, 4, 4, data_format='channels_last',interpolation ='bilinear'))(x_s) 
    
    #x = Upsample(x, [img_height, img_width])

    x_s = Conv2D(nclasses, (1, 1), name='output_layer')(x_s)
    
    x = Activation('softmax')(x_s) 
    """tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    Args:
        from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
        we assume that `y_pred` encodes a probability distribution.
    """    
    model = Model(inputs=base_model.input, outputs=x, name='DeepLabV3_Plus')
    print(f'*** Output_Shape => {model.output_shape} ***')
    return model
