# How to use the Code?<a name="TOP"></a>
===================

- - - - 
# Initializing Model  #

   
    from model_file import unet
    model =unet(input_height=512, input_width=800,classes=2)
    
  

# Compiling Model  #

    from keras.optimizers import Adam
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss=dice_loss, optimizer=opt, metrics=["accuracy"])
    
# Training Model  #

    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    callbacks = [
    ModelCheckpoint('unet_simple_fine.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    history = model.fit(X_train,Y_train, batch_size=5, epochs=20,callbacks=callbacks,validation_data=(X_val,Y_val))
    
# Plotting  Result  #

    from matplotlib import pyplot as plt
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

    

