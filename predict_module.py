import pandas as pd
import numpy as np
import requests
import cv2
import mediapipe as mp
import torch
from PIL import Image
from io import BytesIO
from joblib import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from youtube_transcript_api import YouTubeTranscriptApi

# 모델 및 기타 파일 로드
model = load('model/view_predictor.joblib')
_, _, le_cat = load('model/label_encoders.joblib')
feature_cols = load('model/features.joblib')

# 감성 분석 모델
senti_model_name = "nlp04/korean_sentiment_analysis_kcelectra"
senti_tokenizer = AutoTokenizer.from_pretrained(senti_model_name)
senti_model = AutoModelForSequenceClassification.from_pretrained(senti_model_name)
senti_model.eval()

def sentiment_score(text):
    if not text or pd.isna(text):
        return 0.0
    with torch.no_grad():
        inputs = senti_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = senti_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        try:
            return round(float(probs[2]) * 100, 1)  # Positive
        except IndexError:
            return round(float(probs[1]) * 100, 1)

category_dict = {
    '음식': ['쯔양', '차밥열끼', '먹방', '단골', '아침', '장사', '만들기', '칼로리', '베이글', '곱창', '스테이크', '고기',
           '삼겹살', '성심당', '편의점', '이영자', '라면', '김밥', '치킨', '맛집', '집밥', '떡볶이', '음식', '김치',
           '광어', '만두', '냉면', '철판', '돼지', '요리', '간식', '회식', '술자리', '레시피', '김치찌개'],
    '연예/유명인': ['최화정', '이해리', '개그맨', '강민경', '다비치', '이지혜', '여자', '아이돌', '다나카', '제니',
                '유재석', '핑계고', '조세호', '장영란', '김구라', '김영철', '연예인', '배우', '스타', '출연', '섭외',
                '가수', '노래', '콘서트', '이승철'],
    '교육/공부': ['일차방정식', '이차방정식', '닮음', '인수분해', '지수', '맞춤법', '한국사', '과학', '과외', '수학',
               '수업', '공부', '역사', '공부왕', '수능', '퀴즈', '스터디', '선생님', '시험', '지식', '문제',
               '일차함수', '이차함수', '방정식', '검정고시', '영어', '국어', '한국어', '서울대'],
    '여행/장소': ['두바이', '휴가', '전국', '여행', '투어', '세계', '지하철', '한강', '카페', '코스', '하와이',
                '도쿄', '몽골', '일본', '오사카', '제주', '전주', '제주도', '서울', '민박', '미국', '대만',
                '파리', '스페인', '울릉도', '홍콩'],
    '일상/가족': ['가족', '엄마', '아빠', '남편', '자식', '모녀', '혼자', '하루', '일상', '사람', '아이', '공유',
               '현장', '부부', '가장', '어머니', '조카', '가을', '아들', '결혼식'],
    '콘텐츠/유튜브': ['예능', '시즌', '리뷰', '라이브', '방송', '영상', '채널', '게임', '유튜브', '생방송',
                  '촬영', '콘텐츠', '댓글', '쇼핑'],
    '정치': ['대선', '공약', '안철수', '국회', '정치', '대통령', '선거', '정당', '의원'],
    '경제': ['주식', '비트코인', '코인', '선물', '부자', '투자', '경제', '금융', '광고', '대출', '은행', '시장'],
    '건강/운동': ['운동', '건강', '다이어트', '헬스', '스트레칭', '요가', '체력', '피트니스', '달리기', '근력', '식단'],
    '인간관계/고민': ['연애', '고백', '소개팅', '데이트', '솔로', '고민']
}

# 제목 키워드 기반 분류 함수
def classify_by_keywords(title, keyword_dict):
    for category, keywords in keyword_dict.items():
        for keyword in keywords:
            if keyword in title:
                return category
    return None

# 유튜브 카테고리 + 키워드 기반으로 사용자 카테고리 분류
def map_category(category_id, title, api_key):
    # 유튜브 카테고리 이름 가져오기
    url = f'https://www.googleapis.com/youtube/v3/videoCategories?part=snippet&id={category_id}&regionCode=KR&key={api_key}'
    try:
        res = requests.get(url).json()
        yt_category = res['items'][0]['snippet']['title']
    except:
        yt_category = "기타"

    # 유튜브 카테고리명 → 사용자 카테고리 매핑
    category_map = {
        "영화/애니메이션": "콘텐츠/유튜브",
        "음악": "연예/유명인",
        "엔터테인먼트": "콘텐츠/유튜브",
        "코미디": "콘텐츠/유튜브",
        "인물/블로그": "연예/유명인",
        "게임": "콘텐츠/유튜브",
        "노하우/스타일": "일상/가족",
        "뉴스/정치": "정치",
        "교육": "교육/공부",
        "과학/기술": "교육/공부",
        "스포츠": "건강/운동",
        "자동차": "기타",
        "동물": "기타",
        "여행": "여행/장소"
    }
    mapped_category = category_map.get(yt_category, None)

    # 키워드 기반 보완 분류
    keyword_category = classify_by_keywords(title, category_dict)

    # 최종 우선순위 적용
    return keyword_category or mapped_category or "기타"

def hue_to_color_group(hue_value):
    if 0 <= hue_value < 15 or hue_value >= 345:
        return "빨강 계열"
    elif 15 <= hue_value < 45:
        return "주황/노랑 계열"
    elif 45 <= hue_value < 75:
        return "연두/초록 계열"
    elif 75 <= hue_value < 165:
        return "초록/하늘 계열"
    elif 165 <= hue_value < 255:
        return "파랑/남색 계열"
    elif 255 <= hue_value < 285:
        return "보라 계열"
    elif 285 <= hue_value < 345:
        return "분홍 계열"
    else:
        return "기타"

def analyze_thumbnail(thumbnail_url):
    response = requests.get(thumbnail_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img_np = np.array(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    hue_avg = int(np.mean(hsv[:, :, 0]) * 2)

    # 얼굴 수 검출
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.3) as fd:
        results = fd.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        face_count = len(results.detections) if results.detections else 0

    return hue_to_color_group(hue_avg), face_count, hue_avg

def predict_views(video_id, api_key):
    url = f'https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={api_key}'
    res = requests.get(url).json()
    item = res['items'][0]
    
    title = item['snippet']['title']
    published_at = item['snippet']['publishedAt']
    category_id = item['snippet'].get('categoryId', '')
    thumbnail_url = item['snippet']['thumbnails']['high']['url']
    views = int(item['statistics'].get('viewCount', 0))

    # 게시일 정보
    dt = pd.to_datetime(published_at)
    hour = dt.hour
    weekday = dt.dayofweek

    # 자막 수
    def count_manual_subtitles(video_id):
        ppl = YouTubeTranscriptApi.list_transcripts(video_id)
        manual = [t for t in ppl if not t.is_generated]
        return len(manual)
    
    caption_count = count_manual_subtitles(video_id)

    # 썸네일 분석
    hue_group, face_count, hue_value = analyze_thumbnail(thumbnail_url)

    # 감성 점수
    senti = sentiment_score(title)

    # 카테고리 이름 매핑
    user_category = map_category(category_id, title, api_key)

    # Label Encoding
    if user_category not in le_cat.classes_:
        user_category = '기타'
    cat_encoded = le_cat.transform([user_category])[0]

    # 예측
    X_input = pd.DataFrame([{
        '시간대': hour,
        '요일': weekday,
        '자막수': caption_count,
        '카테고리': cat_encoded,
        'Hue': hue_value,
        '썸네일 얼굴 수': face_count,
        '감성점수': senti
    }])

    pred_log = model.predict(X_input[feature_cols])[0]
    predicted_views = int(np.expm1(pred_log))

    return {
        '제목': title,
        '예측 조회수': predicted_views,
        '실제 조회수': views,
        '카테고리': user_category,
        '시간대': hour,
        '요일': weekday,
        '자막수': caption_count,
        '썸네일 얼굴 수': face_count,
        '감성점수': senti,
        'Hue 그룹': hue_group,
        'Hue 값': hue_value,
        '썸네일 URL': thumbnail_url
    }

#1. 추측 함수
def extract_features_from_video_id(video_id, api_key):
    info = predict_views(video_id, api_key)
    return pd.DataFrame([{
        '시간대': info['시간대'],
        '요일': info['요일'],
        '자막수': info['자막수'],
        '카테고리': le_cat.transform([info['카테고리']])[0],
        'Hue': info['Hue 값'],
        '썸네일 얼굴 수': info['썸네일 얼굴 수'],
        '감성점수': info['감성점수']
    }])

# 2. 예측 함수
def predict_view_count(model, features):
    pred_log = model.predict(features[feature_cols])[0]
    return int(np.expm1(pred_log))

# 3. 시각화 함수
def visualize_result(video_id, features, predicted_view_count, info):
    요일_텍스트 = ['월', '화', '수', '목', '금', '토', '일'][features['요일'].values[0]]
    
    html = f"""
    <div style="background-color: #111; color: white; padding: 20px; border-radius: 10px; max-width: 600px; font-family: Arial, sans-serif;">
        <h2>🎯 예측 조회수: {predicted_view_count:,}회</h2>
        <h3>📌 영상 제목: {info['제목']}</h3>
        <img src="{info['썸네일 URL']}" alt="썸네일 이미지" style="width: 100%; border-radius: 10px; margin-bottom: 20px;"/>
        <ul style="list-style-type: none; padding-left: 0;">
            <li>📽️ <strong>영상 ID:</strong> {video_id}</li>
            <li>👁️ <strong>실제 조회수:</strong> {info['실제 조회수']:,}회</li>
            <li>⏰ <strong>시간대:</strong> {features['시간대'].values[0]}시</li>
            <li>📅 <strong>요일:</strong> {요일_텍스트}</li>
            <li>💬 <strong>자막 수:</strong> {features['자막수'].values[0]}</li>
            <li>🎨 <strong>색상 계열:</strong> {info['Hue 그룹']}</li>
            <li>😃 <strong>썸네일 얼굴 수:</strong> {features['썸네일 얼굴 수'].values[0]}</li>
            <li>🧠 <strong>감성 점수:</strong> {features['감성점수'].values[0]:.2f}</li>
        </ul>
    </div>
    """
    return html