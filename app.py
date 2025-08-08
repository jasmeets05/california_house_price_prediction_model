import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle

col = ['MedInc',
 'HouseAge',
 'AveRooms',
 'AveBedrms',
 'Population',
 'AveOccup',
 'Longitude']
st.title('California Housing Price Prediction')

st.image('''https://jasonbarryteam.com/wp-content/uploads/2021/07/The-Bradbury-Estate-and-the-Rolls-Royce-Ghost-3-1024x613.jpg''')

st.header('Model of housing prices to predict median house values in California',divider = True)

# st.subheader('''User must enter given values to predict prices:
# ['MedInc',
#  'HouseAge',
#  'AveRooms',
#  'AveBedrms',
#  'Population',
#  'AveOccup',
#  'Longitude']
# ''')

st.sidebar.title('Select House Features 🏡')

st.sidebar.image('''https://imageio.forbes.com/blogs-images/amydobson/files/2019/01/630-Nimes-Road-Exterior-1200x722.jpg?format=jfor i in df[col]:pg&height=900&width=1600&fit=bounds''')

#read data
temp_df = pd.read_csv('California.csv')

random.seed(12)

all_values = []
for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var = st.sidebar.slider(f'Select{i} range',int(min_value),int(max_value), 
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.fit_transform([all_values])



with open('house_price_pred_ridge_model.pkl','rb')as f:
    chatgpt = pickle.load(f)


price = chatgpt.predict(final_value)[0]

import time

# st.write(pd.DataFrame(dict(zip(col,all_values)),indexx =  [1]))
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Price')

place = st.empty()
place.image('https://cdn.dribbble.com/userupload/22273130/file/original-b60664b066bca5c94894cf105d1190f8.gif',width = 80)




if price>0:
    
    for i in range(100):
        time.sleep(0.005)
        progress_bar.progress(i + 1)

        
    body = f'Predicted Median House Values Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    st.success(body)

else:
    body = ('Invalid House Features Values')
    st.warning(body)
