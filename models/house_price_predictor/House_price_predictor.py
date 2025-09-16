import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoder
model = joblib.load("models/house_price_predictor/house_predictor.pkl")
encoder = joblib.load("models/house_price_predictor/encoder.pkl")
X_columns = joblib.load("models/house_price_predictor/X_columns.pkl")

# Define categorical and numerical features
categorical_features = ["MSZoning", "LotConfig", "BldgType", "Exterior1st"]
numerical_features = ["MSSubClass", "LotArea", "OverallCond", "YearBuilt", "YearRemodAdd", "BsmtFinSF2", "TotalBsmtSF"]

def house_price_predictor():
    st.write("Fill in the property details below to predict the house price.")
    
    # Use st.container() to group inputs, which can help with layout
    with st.container():
        # User inputs for numerical features
        num_inputs = {}
        for col in numerical_features:
            num_inputs[col] = st.number_input(f"{col}", value=0)

        # User inputs for categorical features
        cat_inputs = {}
        cat_inputs['MSZoning'] = st.selectbox("MSZoning", ['RL', 'RM', 'C (all)', 'FV', 'RH'])
        cat_inputs['LotConfig'] = st.selectbox("LotConfig", ['Inside', 'FR2', 'Corner', 'CulDSac', 'FR3'])
        cat_inputs['BldgType'] = st.selectbox("BldgType", ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twns'])
        cat_inputs['Exterior1st'] = st.selectbox("Exterior1st", [
            'VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd',
            'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn', 'Stone', 'ImStucc',
            'CBlock'
        ])

    # Predict button
    if st.button("Predict Price"):
        # Create input DataFrame inside the if block
        user_data = {**num_inputs, **cat_inputs}
        input_df = pd.DataFrame([user_data])

        # Process categorical features with OneHotEncoder
        encoded_cats = encoder.transform(input_df[categorical_features])
        encoded_df = pd.DataFrame(
            encoded_cats,
            columns=encoder.get_feature_names_out(categorical_features),
            index=input_df.index
        )

        # Combine numeric + categorical
        final_input = pd.concat([input_df[numerical_features], encoded_df], axis=1)

        # Reindex to match training features
        final_input = final_input.reindex(columns=X_columns, fill_value=0)

        # Make and display the prediction
        prediction = model.predict(final_input)
        st.markdown(f"<h4>Predicted House Price: </h4><h3 style='text-align:center;'>{prediction[0]:,.2f}</h3>",unsafe_allow_html=True)

