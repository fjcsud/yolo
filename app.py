from io import BytesIO

import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(page_title="YOLO 사진 탐지", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="YOLO 모델을 불러오는 중입니다...")
def load_model():
    return YOLO("yolo26n.pt")


st.title("🔎 YOLO 사진 객체 탐지")
st.write("사진을 업로드하거나 카메라로 촬영하면 YOLO가 이미지 속 객체를 찾아 표시합니다.")
st.caption("첫 실행 시 모델 가중치를 다운로드하므로 잠시 기다려 주세요.")

input_mode = st.radio(
    "사진 입력 방법",
    ["사진 업로드", "카메라 촬영"],
    horizontal=True,
)

if input_mode == "사진 업로드":
    photo = st.file_uploader(
        "분석할 사진을 선택하세요",
        type=["jpg", "jpeg", "png", "webp"],
    )
else:
    photo = st.camera_input("객체가 보이도록 사진을 촬영하세요")

confidence = st.slider(
    "탐지 신뢰도 기준",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="이 기준보다 낮은 신뢰도의 탐지 결과는 표시하지 않습니다.",
)

if photo is None:
    st.info("시작하려면 사진을 업로드하거나 카메라로 촬영하세요.")
else:
    image = Image.open(BytesIO(photo.getvalue())).convert("RGB")
    model = load_model()

    with st.spinner("사진을 분석하고 있습니다..."):
        result = model.predict(image, conf=confidence, verbose=False)[0]

    annotated_image = Image.fromarray(result.plot()[..., ::-1].copy())
    boxes = result.boxes
    detections = []

    if boxes is not None:
        for coordinates, class_id, score in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.cls.cpu().tolist(),
            boxes.conf.cpu().tolist(),
        ):
            detections.append(
                {
                    "객체": result.names[int(class_id)],
                    "신뢰도": round(score, 3),
                    "위치 (x1, y1, x2, y2)": ", ".join(
                        f"{coordinate:.1f}" for coordinate in coordinates
                    ),
                }
            )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("탐지 결과")
        st.image(annotated_image, use_container_width=True)
    with right:
        st.subheader("탐지 요약")
        st.metric("탐지된 객체", len(detections))
        if detections:
            st.dataframe(detections, use_container_width=True, hide_index=True)
        else:
            st.info("이 기준 이상의 신뢰도를 가진 객체를 찾지 못했습니다.")
