# 🚗2023ESWContest_mobility_명장_6056

## 🕵️‍♀️T-VAR(Traffic signal Violation Automatic Reporting system)
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/124027423/aa2c9cd3-94e0-477c-998d-246843566b75)

### 📜작품 소개
본 시스템은 차량 내 빌트인 캠을 활용한 신호 위반 자동 단속 시스템으로, 빌트인 캠의 영상과 라즈베리파이, GPS 센서를 이용한다.
앞 차량을 빌트인 캠 및 나의 GPS 정보를 이용해 거리를 측정하고, 나와 신호등까지의 거리를 기준으로 신호 위반을 판단한다.
최종적으로 판단된 정보를 서버에 저장 및 Application으로 전송해 자동으로 양식을 완성해 준다.

## 시연 영상
<img src="https://img.shields.io/badge/youtube-FF0000?style=for-the-badge&logo=youtube&logoColor=White">
https://myoungjang.site/youtube

## 💁‍♂️팀 소개 및 역할
| Position    | Name&nbsp;&nbsp;   | GitHub | Email | Role |
|:-------------:|:--------:|--------|-------|------|
| **Team Leader** | 김영휘 | https://github.com/dudgnl5209| dudgnl5209@naver.com| 기획 및 개발 총괄, 웹 스트리밍 통신 구현, 위반 데이터 저장소 관리,Reverse Geocoding 통합 처리, 공공 데이터 수집 및 가공     |
| **Team member** | 김우주 | https://github.com/pupukii   | wj3507@naver.com    | 객체 추적 모델 설계 및 구현, Image Processing, 신호 위반 판단 Logic, 신호 위반 관련 법규 분석     |
| **Team member** | 신유재 | https://github.com/Uj710     | yujae710@naver.com  | Server 통신 관리, Application 개발, UI디자인, 저장소와 Application 간의 통신 개발     |
| **Team member** | 한현준 |        | alqp201@gmail.com   | 객체 추적 모델 설계 및 구현, Image Processing, 신호 위반 판단 Logic 구현, 위반 데이터 후처리     |
| **Team member** | 오준혁 |        | stephan330@naver.com| Custom data 학습 파일 생성, 공공 데이터 수집 및 가공, 모듈 Case 설계 및 안전성 검증, 데이터 증강 알고리즘 개발     |

### 📌영역별 소개
1. Built-in Cam Module : 상시 촬영을 진행하며, 신호등과의 거리가 50m 안으로 들어올 때부터 서버에 실시간으로 스트림을 진행한다
2. Server : 실시간으로 전송받은 영상 정보에 대해 객체 인식을 진행하고 인식된 객체와 나의 거리 파악, 서울시 공공데이터를 기준으로 신호등과 나의 거리를 파악을 통해
신호 위반 판단을 진행한다.
3. Cloud Storage : 신호 위반 판단으로 판정날 경우 차번호, 위반날짜, 위반장소, 위반 항목을 txt파일로 저장하고, 이에 해당하는 위반 영상을 같이 저장 및 관리한다.
( 이때 저장소는 48시간이 넘은 데이터는 자동으로 삭제하도록 한다)
4. Application : 사용자가 저장소에 로그인 하여 저장된 정보들을 리스트로 받아온다. 신호 위반이라 판단된 정보를 확인할 수 있고 원하는 경우 자동으로 양식을 생성해준다.
5. API : 위치 좌표를 Reverse Geo Coding을 이용해 도로명 주소로 바꿔준다.


### 📁파일 구성
----------------
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/124027423/df1933c6-b62a-4ed1-b564-d97588a0dcc2)

### 👨‍💻영역별 개발 환경
--------------------
#### 1. 📷Built-in-Cam
>###### OS
<img src="https://img.shields.io/badge/Raspbian_OS-A22846?style=for-the-badge&logo=Raspberry Pi&logoColor=Red">

>###### Language
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">

>###### Framework
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=White"> <img src="https://img.shields.io/badge/flask-3481FE?style=for-the-badge&logo=flask&logoColor=White">

#### 2. 🛜Server
>###### IDE
<img src="https://img.shields.io/badge/googlecolab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white">

>###### Language
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">

>###### Framework
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=White"> <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=White"> <img src="https://img.shields.io/badge/yolo-00FFFF?style=for-the-badge&logo=yolo&logoColor=white">

#### 3. Application
>###### IDE
<img src="https://img.shields.io/badge/androidstudio-3DDC84?style=for-the-badge&logo=androidstudio&logoColor=white">

>###### Language
<img src="https://img.shields.io/badge/dart-0175C2?style=for-the-badge&logo=dart&logoColor=white">

>###### Framework
<img src="https://img.shields.io/badge/flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white">

>###### API
<img src="https://img.shields.io/badge/googledrive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white"> <img src="https://img.shields.io/badge/Kakaomaps-FFCD00?style=for-the-badge&logo=googlemaps&logoColor=white">

## 🔍각 영역별 구성도
### 📷Built - in Cam Module 구성
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/124027423/ced773f5-92f7-4f61-89d0-5ff33362b445)

### 🛜Server 구성 및 상황별 판단 Logic
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/124027423/b0974b6d-9655-4c19-b5e5-d732a5097622)

### 📱Application 구성
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/124030468/359f16c2-7100-4dd5-b490-d9db40d1fe59)

### 📲Application UI GUIDE
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/124030468/e63015f9-98e6-4c62-8250-0e61f8a0263b)


