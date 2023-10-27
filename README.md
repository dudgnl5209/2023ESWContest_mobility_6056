# 2023ESWContest_mobility_6056
# T-VAR( Traffic signal Violation Automatic Reporting system )
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/116995224/ecdb9471-a84f-4753-afbc-9ab51624df65)
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
 ![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/116995224/0c477f07-1ad8-4135-81a2-724f4c34921c)
# Built - in Cam Module 구성
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/116995224/6af9932a-5424-4409-94ea-d80bb5c572fd)
# Server 구성 및 상황별 판단 Logic
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/116995224/bc0abb0f-d527-4996-b297-d78058aa362e)
# Application 구성
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/116995224/a498fc38-25d1-4e8b-af8a-4ee13972a9c9)
# Application 사용자 가이드

# 영역별 개발 환경
![image](https://github.com/dudgnl5209/2023ESWContest_mobility_6056/assets/116995224/1dc53eff-8f67-4eda-bc12-3ea82ed63f46)
