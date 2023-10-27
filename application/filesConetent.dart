import 'package:flutter/material.dart';
import 'package:chewie/chewie.dart';
import 'package:video_player/video_player.dart';
import 'package:http/http.dart' as http;
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'dart:convert';

class MapViewScreen extends StatefulWidget {
  final String latitude;
  final String longitude;

  MapViewScreen({required this.latitude, required this.longitude});

  @override
  _MapViewScreenState createState() => _MapViewScreenState();
}

class _MapViewScreenState extends State<MapViewScreen> {
  bool _isSatelliteView = false;

  @override
  Widget build(BuildContext context) {
    double lat = double.parse(widget.latitude);
    double lon = double.parse(widget.longitude);

    LatLng initialPosition = LatLng(lat, lon);

    return Scaffold(
      appBar: AppBar(
        title: Text(
          '지도 보기',
          style: TextStyle(
            color: _isSatelliteView ? Colors.white : Colors.black,
          ),
        ),
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: [
          IconButton(
            icon: Icon(Icons.map),
            onPressed: () {
              setState(() {
                _isSatelliteView = !_isSatelliteView;
              });
            },
          ),
        ],
        iconTheme: IconThemeData(color: _isSatelliteView ? Colors.white : Colors.black),
      ),
      extendBodyBehindAppBar: true,
      body: GoogleMap(
        initialCameraPosition: CameraPosition(
          target: initialPosition,
          zoom: 15.0,
        ),
        mapType: _isSatelliteView
            ? MapType.satellite
            : MapType.normal,
        markers: Set<Marker>.of([
          Marker(
            markerId: MarkerId('1'),
            position: LatLng(lat, lon),
            infoWindow: InfoWindow(
              title: '위반 위치',
            ),
          ),
        ]),
      ),
    );
  }
}


class FileContentScreen extends StatelessWidget {
  //GoogleSignInAccount user;
  final String content;
  final String fileName;
  final String mp4FileName;
  final String mp4FileId;
  final String thumbnailUrl;

  FileContentScreen({
    // required this.user,
    required this.content,
    required this.fileName,
    required this.mp4FileName,
    required this.mp4FileId,
    required this.thumbnailUrl,
  });

  Future<String> getAddress(String lat, String lon) async {
    final apiKey = 'AIzaSyCtLU3xO0Tn8aEkRB1YheMn8kwybz70Km0';
    final url = 'https://maps.googleapis.com/maps/api/geocode/json?latlng=$lat,$lon&key=$apiKey&language=ko';

    final response = await http.get(Uri.parse(url));

    if (response.statusCode == 200) {
      final Map<String, dynamic> data = json.decode(response.body);
      final results = data['results'];
      if (results.isNotEmpty) {
        return results[0]['formatted_address'];
      }
      else {
        // 오류 처리
        print('주소를 가져올 수 없습니다. 상태 코드: ${response.statusCode}');
      }
    }
    return 'Could not fetch address';
  }


  @override
  Widget build(BuildContext context) {
    String displayFileName = fileName.split('.').first;

    List<String> contentLines = content.split('\n');

    List<String> tmp = contentLines[4].split(': ')[1].split(', ');

    print(tmp[0]);
    print(tmp[1]);

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Color(0xFFF4F7FF),
        elevation: 0,
        title: Padding(
          padding: EdgeInsets.only(top: 24.0),
          child: Text(
            '제보 양식서 확인',
            style: TextStyle(fontSize: 20.0, color: Colors.black),
          ),
        ),
        leading: Padding(
          padding: EdgeInsets.only(top: 15.0),
          child: IconButton(
            icon: Icon(Icons.arrow_back),
            onPressed: () {
              Navigator.pop(context);
            },
            iconSize: 30.0,
            color: Colors.black,
          ),
        ),
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(height: 30.0),
            GestureDetector(
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => ChewieListItem(
                      videoPlayerController: VideoPlayerController.network(
                        'https://drive.google.com/uc?id=$mp4FileId',
                      ),
                      looping: false,
                    ),
                  ),
                );
              },
              child: Center(
                child: Image.network(
                  thumbnailUrl,
                  fit: BoxFit.fitWidth,
                  width: 300,
                ),
              ),
            ),
            SizedBox(height: 30.0),
            Container(
              padding: EdgeInsets.only(top: 8.0, bottom: 8.0),
              margin: EdgeInsets.all(8.0),
              child: Column(
                children: [
                  ListView.builder(
                    shrinkWrap: true,
                    itemCount: 4,
                    itemBuilder: (BuildContext context, int index) {
                      List<String> parts = contentLines[index].split(' : ');

                      return Column(
                        children: [
                          Row(
                            children: [
                              Expanded(
                                flex: 2,
                                child: Container(
                                  padding: EdgeInsets.only(left: 20.0, top: 5.0, bottom: 5.0),
                                  child: Text(
                                    parts[0],
                                    style: TextStyle(
                                      fontSize: 16.0,
                                      fontWeight: index.isOdd ? FontWeight.bold : FontWeight.bold,
                                    ),
                                  ),
                                ),
                              ),
                              SizedBox(width: 30.0),
                              Expanded(
                                flex: 5,
                                child: Container(
                                  padding: EdgeInsets.only(top: 5.0, bottom: 5.0),
                                  child: Text(
                                    parts[1],
                                    style: TextStyle(
                                      fontSize: 16.0,
                                      fontWeight: index.isOdd ? FontWeight.bold : FontWeight.bold,
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                          Divider(
                            color: Colors.grey,
                            thickness: 1.0,
                          ),
                        ],
                      );
                    },
                  ),

                  Row(
                    children: [
                      Expanded(
                        flex: 2,
                        child: Container(
                          padding: EdgeInsets.only(left: 20.0, top: 5.0, bottom: 5.0),
                          child: Text(
                            '위반위치',
                            style: TextStyle(
                              fontSize: 16.0,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ),
                      SizedBox(width: 30.0),
                      Expanded(
                        flex: 5,
                        child: FutureBuilder<String>(
                          future: getAddress(tmp[0], tmp[1]),
                          builder: (BuildContext context, AsyncSnapshot<String> snapshot) {
                            if (snapshot.connectionState == ConnectionState.waiting) {
                              return CircularProgressIndicator(); // 로딩 중인 동안 표시될 위젯
                            } else if (snapshot.hasError) {
                              return Text('Error: ${snapshot.error}');
                            }  else {
                              String addressWithoutCountry = snapshot.data!.replaceAll('대한민국', '').trim();

                              return Container(
                                padding: EdgeInsets.only(top: 5.0, bottom: 5.0),
                                child: Text(
                                  addressWithoutCountry,
                                  style: TextStyle(
                                    fontSize: 16.0,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              );
                            }
                          },
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: 6),

                  Container(
                    width: 180, // 버튼의 너비 조정
                    height: 45, // 버튼의 높이 조정
                    child: ElevatedButton(
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => MapViewScreen(
                              latitude: tmp[0],
                              longitude: tmp[1],
                            ),
                          ),
                        );
                      },
                      style: ElevatedButton.styleFrom(
                        primary: Color(0x5463d6),
                        onPrimary: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
                      ),
                      child: Align(
                        alignment: Alignment.center, // 텍스트와 이미지를 가운데로 정렬합니다.
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Image.asset(
                              'assets/map.png',
                              width: 30,
                              height:30,
                            ),
                            SizedBox(width: 10),
                            Text(
                              '지도에서 보기',
                              textAlign: TextAlign.center,
                            ),
                          ],

                        ),

                      ),

                    ),

                  ),
                  SizedBox(height: 2),
                  Divider(
                    color: Colors.grey,
                    thickness: 1.0,
                  ),

                  Padding(
                    padding: EdgeInsets.only(left: 20.0, top: 5.0),
                    child: Align(
                      alignment: Alignment.centerLeft,
                      child: Text(
                        '신고내용',
                        style: TextStyle(
                          fontSize: 16.0,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(height: 10.0),
                  TextField(
                    decoration: InputDecoration(
                      hintText: '사용자 입력란',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: 30.0),
            Center(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: ElevatedButton(
                  onPressed: () {
                    showDialog(
                      context: context,
                      builder: (BuildContext context) {
                        return AlertDialog(
                          title: Text('제보 완료'),
                          content: Text('제보가 완료되었습니다.'),
                          actions: [
                            TextButton(
                              onPressed: () {
                                Navigator.of(context).pop();
                              },
                              child: Text('확인'),
                            ),
                          ],
                        );
                      },
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    primary: Color(0x5463d6),
                    onPrimary: Colors.white,
                    padding: EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                  child: Text(
                    '제보하기',
                    style: TextStyle(fontSize: 20.0),
                  ),
                ),
              ),
            ),SizedBox(height: 30.0),
          ],
        ),
      ),
    );
  }
}

class ChewieListItem extends StatefulWidget {
  final VideoPlayerController videoPlayerController;
  final bool looping;

  ChewieListItem({
    required this.videoPlayerController,
    this.looping = false,
  });

  @override
  _ChewieListItemState createState() => _ChewieListItemState();
}

class _ChewieListItemState extends State<ChewieListItem> {
  ChewieController? _chewieController;

  @override
  void initState() {
    super.initState();
    _chewieController = ChewieController(
      videoPlayerController: widget.videoPlayerController,
      aspectRatio: 16 / 9,
      autoInitialize: true,
      looping: widget.looping,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: PreferredSize(
        preferredSize: Size.fromHeight(80.0), // 이 부분에서 높이를 조정합니다.
        child: AppBar(
          backgroundColor: Color(0xFFF4F7FF),
          elevation: 0,
          title: Padding(
            padding: EdgeInsets.only(top: 24.0),
            child: Text(
              '제보 영상',
              style: TextStyle(fontSize: 20.0, color: Colors.black),
            ),
          ),
          leading: Padding(
            padding: EdgeInsets.only(top: 15.0),
            // 상단과 왼쪽에 각각 20.0만큼의 패딩을 추가합니다.
            child: IconButton(
              icon: Icon(Icons.arrow_back), // 뒤로가기 아이콘
              onPressed: () {
                // 뒤로가기 버튼이 눌렸을 때 실행되는 함수
                Navigator.pop(context);
              },
              iconSize: 30.0, // 아이콘 크기를 30으로 설정
              color: Colors.black, // 아이콘 색상 설정
            ),
          ),
        ),
      ),
      body: Padding(
        padding: EdgeInsets.only(bottom: 100), // 조정하고 싶은 위치값으로 변경하세요
        child: Chewie(
          controller: _chewieController!,
        ),
      ),

    );
  }

  @override
  void dispose() {
    super.dispose();
    _chewieController!.dispose();
  }
}