import streamlit as st
import pickle
import numpy as np

st.title('Clasificador Diabetes')
st.write("Esta es una aplicación simple que toma la edad y el nivel de glucosa para predecir la diabates")
arch_modelo = open('modelo.pickle', 'rb')
salidas_modelo = open('salidas_modelo.pickle', 'rb')
modelo = pickle.load(arch_modelo)
salidas = pickle.load(salidas_modelo)

arch_modelo.close()
salidas_modelo.close()

with st.form('entradas_usuario'):    
    edad = st.number_input("Edad", min_value=0)
    glucosa = st.number_input("Nivel de glucosa", min_value=0)
    entradas = [edad, glucosa]
    
    st.write(f"""La entrada del usuario es {entradas}""".format())
    submit = st.form_submit_button(label='Enviar')

if submit:       
    datos_prueba = np.array([[edad, glucosa]])         
    prediccion = modelo.predict(datos_prueba)
    print("\n", prediccion)
    salida_predicha = "Si" if salidas[prediccion][0] == 1 else "No"
    st.write(f"Sr Usuario probablemente ud {salida_predicha} tiene diabetes")




