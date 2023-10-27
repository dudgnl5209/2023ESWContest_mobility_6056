# 2023ESWContest_mobility_명장_6056
# T-VAR( Traffic signal Violation Automatic Reporting system )
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/124027423/aa2c9cd3-94e0-477c-998d-246843566b75)

# 작품 소개
본 시스템은 차량 내 빌트인 캠을 활용한 신호 위반 자동 단속 시스템으로, 빌트인 캠의 영상과 라즈베리파이, GPS 센서를 이용한다.
앞 차량을 빌트인 캠 및 나의 GPS 정보를 이용해 거리를 측정하고, 나와 신호등까지의 거리를 기준으로 신호 위반을 판단한다.
최종적으로 판단된 정보를 서버에 저장 및 Application으로 전송해 자동으로 양식을 완성해 준다.
# 영역별 소개
1. Built-in Cam Module : 상시 촬영을 진행하며, 신호등과의 거리가 50m 안으로 들어올 때부터 서버에 실시간으로 스트림을 진행한다
2. Server : 실시간으로 전송받은 영상 정보에 대해 객체 인식을 진행하고 인식된 객체와 나의 거리 파악, 서울시 공공데이터를 기준으로 신호등과 나의 거리를 파악을 통해
신호 위반 판단을 진행한다.
3. Cloud Storage : 신호 위반 판단으로 판정날 경우 차번호, 위반날짜, 위반장소, 위반 항목을 txt파일로 저장하고, 이에 해당하는 위반 영상을 같이 저장 및 관리한다.
( 이때 저장소는 48시간이 넘은 데이터는 자동으로 삭제하도록 한다)
4. Application : 사용자가 저장소에 로그인 하여 저장된 정보들을 리스트로 받아온다. 신호 위반이라 판단된 정보를 확인할 수 있고 원하는 경우 자동으로 양식을 생성해준다.
5. API : 위치 좌표를 Reverse Geo Coding을 이용해 도로명 주소로 바꿔준다.
# 파일 구성
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/124027423/df1933c6-b62a-4ed1-b564-d97588a0dcc2)

# Built - in Cam Module 구성
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/124027423/ced773f5-92f7-4f61-89d0-5ff33362b445)

# Server 구성 및 상황별 판단 Logic
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/124027423/b0974b6d-9655-4c19-b5e5-d732a5097622)

# Application 구성
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/124030468/359f16c2-7100-4dd5-b490-d9db40d1fe59)

# Application UI GUIDE
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/124030468/87a8a839-cb08-4385-81be-fb9f85d7391f)

# 영역별 개발 환경
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/124027423/d3907337-6d2d-4e1c-8dc3-8b9684ae827a)

