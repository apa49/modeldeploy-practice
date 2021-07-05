import streamlit as st
st.title("Our Streamlit Project")
from PIL import Image
st.subheader(" This is a subheader")
image=Image.open("D:\streamlit\datacsience1.png")
st.image(image,use_column_width=True)
st.write("writing a text here")
st.markdown("this is markdown cell")
st.success("congrats")
st.info("This is an info")
st.warning("be carefull")
st.error("oops you into an error")
st.help(range)
import numpy as np
import pandas as pd
data=np.random.rand(10,20)
st.dataframe(data)
st.text("---"*100)
df=pd.DataFrame(np.random.rand(10,20),columns=('col %d' % i for i in range(20)))
st.dataframe(df.style.highlight_max(axis=1))
chart_data=pd.DataFrame(np.random.randn(20,3),columns=['a','b','c'])
st.line_chart(chart_data)
st.text("---"*100)
st.area_chart(chart_data)
st.bar_chart(chart_data)
import matplotlib.pyplot as plt
arr=np.random.normal(1,1,size=100)
plt.hist(arr,bins=20)
st.pyplot()

if st.button("say hello"):
    st.write("hello is here")
else:
    st.write("why are u here")    
genre=st.radio('what is your favourite genre',('Comedy','Drama'))    
option=st.multiselect("how was ur night",('Fantastic',"Awesome"))
st.write("Your",option)
age=st.slider("how old are you?",0,100,80)
st.write("Your",age)
values=st.slider("Range",0,200,(15,80))
st.write("Your",values)
number=st.number_input("Input Number")
st.write("NuMber",number)
upload_file=st.file_uploader("Choose a file",type='csv')
if upload_file is not None:
    data=pd.read_csv(upload_file)
    st.write(data)
    st.success("success")
else:
    st.error("file uploaded wrong")    
 
add=st.sidebar.selectbox("what is favourite colour",('Computer','Maths'))
st.balloons()

