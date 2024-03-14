import pandas as pd
# $WIPE_BEGIN
# from tensorflow.keras.models import load_model
from keras.models import load_model
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# from taxifare.ml_logic.registry import load_model
# from taxifare.ml_logic.preprocessor import preprocess_features
# $WIPE_END
import train_model.nural_train.train_gen as tn_gn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.staticfiles import StaticFiles
import os
import time
import train_model.deploy_fourier_model.predict  as b_pr

def list_pictures_in_folders(folder_path):
    folders_list = [item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]
    pictures_in_folders = []

    for folder in folders_list:
        folder_path_full = os.path.join(folder_path, folder)
        all_files = os.listdir(folder_path_full)
        pictures = [file for file in all_files if file.endswith(('.jpg', '.png', '.jpeg'))]  # Add other picture formats if needed
        pictures_in_folders.append({folder: pictures})

    return pictures_in_folders
    

# Replace 'your_folder_path_here' with the actual path of your folder
folder_path = 'train_model/train_bearing_image/'
pictures_list = list_pictures_in_folders(folder_path)
# print (pictures_list)


app = FastAPI()


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.mount("/good", StaticFiles(directory="train_model/train_bearing_image/good"), name="static")
app.mount("/fail", StaticFiles(directory="train_model/train_bearing_image/fail"), name="static")

app.state.model  = load_model('Cloud_model.h5')


app.state.model_img  = load_model('train_model/cnnv4good.h5')

app.state.model_train_normal  = load_model('train_model/nural_train/Cloud_modeltrain_002.h5')

app.state.model_train_crack  = load_model('train_model/nural_train/Cloud_modeltrain_crack_001.h5')

print("Model loaded successfully")


merged_data = pd.read_csv("new_preprocces.csv" ,index_col=0)
train_bigdata = pd.read_csv("train_model/nural_train/bigdata.csv" ,index_col=0)
train = merged_data[800:950]
test = merged_data[800:950]
test_norm = merged_data[600:750]
test_start = merged_data[730:880]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
X_test_norm = scaler.transform(test_norm)
X_test_start = scaler.transform(test_start)
scaler_filename = "scaler_data"
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
X_test_norm = X_test_norm.reshape(X_test_norm.shape[0], 1, X_test_norm.shape[1])
X_test_start = X_test_start.reshape(X_test_start.shape[0], 1, X_test_start.shape[1])
print("Test data shape:", X_test.shape)
X_pred = app.state.model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['Threshold'] = 0.05
scored['waring'] = scored['Loss_mae'] < scored['Threshold']
print (scored.head())

def predict_image(image_path, model, class_labels, target_size=(143, 143)):
    """
    Load an image, preprocess it, and perform a prediction using the specified model.

    Parameters:
    - image_path: str, path to the image file.
    - model_path: str, path to the trained Keras model.
    - class_labels: list, a list of class labels in the order they were encoded.
    - target_size: tuple, the target size of the image input for the model.

    Returns:
    - prediction_label: str, the label of the predicted class.
    - prediction_probability: float, the probability of the predicted class.
    """
    # Load and preprocess the image

    image = tf.keras.preprocessing.image.load_img(image_path, color_mode='rgba', target_size=target_size) # Ensure to load with RGBA
    image_array = tf.keras.preprocessing.image.img_to_array(image)

    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    print (image_path)

    predictions = model.predict(image_array)

    # Find the predicted class
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    prediction_label = class_labels[predicted_class_idx]
    prediction_probability = np.max(predictions)

    return prediction_label, prediction_probability , image
@app.get("/predict")
def predict(test_name: str):      # 1
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    # $CHA_BEGIN

    # 💡 Optional trick instead of writing each column name manually:
    # locals() gets us all of our arguments back as a dictionary
    # https://docs.python.org/3/library/functions.html#locals
    train = merged_data[800:950]
    test = merged_data[800:950]
    test_norm = merged_data[600:750]
    test_start = merged_data[730:880]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train)
    X_test = scaler.transform(test)
    X_test_norm = scaler.transform(test_norm)
    X_test_start = scaler.transform(test_start)
    scaler_filename = "scaler_data"
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    print("Training data shape:", X_train.shape)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    X_test_norm = X_test_norm.reshape(X_test_norm.shape[0], 1, X_test_norm.shape[1])
    X_test_start = X_test_start.reshape(X_test_start.shape[0], 1, X_test_start.shape[1])
    if test_name == "failer_test":
        print ("fail")
    elif test_name == "norm_test":
        X_test=X_test_norm
    elif test_name == "between_test":
        X_test=X_test_start



    X_pred = app.state.model.predict(X_test)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred, columns=test.columns)
    X_pred.index = test.index

    scored = pd.DataFrame(index=test.index)
    Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
    scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
    scored['Threshold'] = 0.05
    scored['waring'] = scored['Loss_mae'] > scored['Threshold']
    res = scored.head().to_dict()
    warnings = np.array(scored['waring'])
    num_true = np.sum(warnings)
    threshold = 0.6 * warnings.size
    result=bool( num_true > threshold)

    gg = X_test.reshape(X_test.shape[0], X_test.shape[2])

    res =  {"warning":result , "vib" :list (gg[:,0]) }
    print (res)
    return dict(results=res )
    # $CHA_END

@app.get("/img_pridiction")
async def img_pre(cat , filename):
   
    class_labels = [ 'fail','good']  # Replace with your actual class labels
    image_path = f'train_model/train_bearing_image/{cat}/{filename}'

    predicted_label, predicted_probability , imagess = predict_image(image_path, app.state.model_img, class_labels)
    print(f'Predicted Label: {predicted_label}, Probability: {predicted_probability}')
    print (cat , filename)
    return dict(results={"lable" :str(predicted_label), "probability":str(predicted_probability ) })
    # $CHA_END


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END

@app.get("/test_list")
def root():
    res = ["failer_test" , "norm_test","between_test"]
    return dict(results=res)
    # $CHA_END

@app.get("/img_list")
def img_list():
    
    res = {"good": pictures_list[0]["good"],"fail" : pictures_list[1]["fail"]}
    return dict(results=res)
    # $CHA_END

@app.get("/train_prob")
def train_prop(speed  , pos , sev , types ):
    print (">>>>>>" ,(int(speed),int(pos), int(sev) ,types))
    df= tn_gn.data_df(int(speed),int(pos), int(sev) ,types)
    print (df.shape)
    df =df[0:1]
    # print ("train prop test " , df["crack"])
    df = df.drop(["fualt_type","fauly_sev","fault_dis"], axis= 1 )
    g =np.array (df[0:1])
    # print ("train prop test" ,g)
    gg = train_bigdata 
    main_data = gg.drop(["fualt_type","fauly_sev","fault_dis"], axis= 1 )

    # scaler3 = MinMaxScaler()
    # scaler3.fit_transform(main_data)
    scaler3 = joblib.load("train_model/nural_train/scaler_datax4545") 
    X_train = scaler3.transform(main_data[3100:3101])
    # X_train1 = scaler3.transform(gg)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

    X_pred = app.state.model_train_normal.predict(X_train)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred, columns=df.columns)
    X_pred.index = df.index

    scored = pd.DataFrame()
    Xtest = X_train.reshape(X_train.shape[0], X_train.shape[2])
    scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
    scored['Threshold'] = 0.09
    scored['good'] = scored['Loss_mae'] < scored['Threshold']
    scored.head()
    print (">>>: ",scored[0:1]['good'][0])

    if bool (scored[0:1]['good'][0]) == False:
        df1 = df.drop(["feq"], axis= 1 )
        g1 =np.array (df1[0:1])
        # gg = train_bigdata [train_bigdata["fualt_type"].isnull()]
        # main_.ata = gg.drop(["fualt_type","fauly_sev","fault_dis" , "feq"], axis= 1 )
        scaler2 = joblib.load("train_model/nural_train/scaler_data_x111") 
        # scaler2.fit_transform(main_data)
        X_train2 = scaler2.transform(g1)
        print ("testing shape ")
        print (X_train2.shape)

        scaler = joblib.load("train_model/nural_train/scaler_data_y111") 
        data = scaler.inverse_transform(app.state.model_train_crack.predict(X_train2)) 
        print (data)
        scored["fail"] = data.tolist()

    ress =scored.to_dict()
    res = {"data" :ress}
    return dict(results=res)


@app.get("/user_op")
def user_op():
    op = b_pr.get_user_options()
    res = {"op": op }
    return dict(results=res)
    # $CHA_ENDs

@app.get("/get_vibration_dataframe")
def get_vibration_dataframe():
    df = b_pr.get_vibration_dataframe()
    res = {"df":df.to_dict()  }
    return dict(results=res)
    # $CHA_ENDs

@app.get("/user_opselect")
def user_op_select(option):
    print(option)
    data = b_pr.get_signal_for_plotting(option)
    print ("the data : " , data)
    res = {"op": data.tolist()}
    return dict(results=res)
    # $CHA_ENDs
@app.get("/b_pricit")
def b_pricit(option):
    print(option)
    data = b_pr.predict(option)
    print ("the data : " , data)
    res = {"op": data}
    return dict(results=res)
    # $CHA_ENDs