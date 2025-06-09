import streamlit as st
import pandas as pd
import joblib
from predict_module import extract_features_from_video_id, predict_view_count, visualize_result, predict_views

api_key = "AIzaSyAgkZQp9EqA6N49J7TCh6Q40mWyVIGBit8"
model = joblib.load("model/view_predictor.joblib")

st.title("🎬 YouTube 조회수 예측기")

video_id = st.text_input("YouTube 영상 ID를 입력하세요:")

if st.button("예측 시작"):
    try:
        # ❶ info = 영상 전체 정보 포함
        info = predict_views(video_id, api_key)

        # ❷ features만 추출
        features = extract_features_from_video_id(video_id, api_key)

        # ❸ 예측
        predicted = predict_view_count(model, features)

        # ❹ 시각화할 때 info도 넘김
        html = visualize_result(video_id, features, predicted, info)
        st.components.v1.html(html, height=1000)

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
