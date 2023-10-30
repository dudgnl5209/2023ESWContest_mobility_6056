from google.colab import files
import datetime
drive.mount('/content/gdrive')
file_path1 = '/...'   #구글 드라이브 경로 넣기 /content/gdrive/...
def save_result_to_drive(result):                   #result : 차량번호, 위반장소, 위반위치
    now = datetime.datetime.now()
    file_name = now.strftime("%Y%m%d_%H%M") + '.txt'
    file_path = file_path1 + file_name
    date = datetime.datetime.now().strftime("%Y년 %m월 %d일")
    hour = datetime.datetime.now().strftime("%H시 %M분")
    with open(file_path, 'w') as file:
        result = result.split(',')
        file.write('발생일자 : '+ date +'\n')
        file.write('발생시각 : ' + hour + '\n')
        file.write('차량번호 : ' + result[0] + '\n')
        file.write('위반항목 : 신호위반\n')
        file.write('위반위치 : ' + result[1] + ',' + result[2] + '\n')
    file.close()

