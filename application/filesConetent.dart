import 'package:flutter/material.dart';

import 'package:chewie/chewie.dart';
import 'package:video_player/video_player.dart';

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

  @override
  Widget build(BuildContext context) {
    String displayFileName = fileName.split('.').first;

    List<String> contentLines = content.split('\n');

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
                    itemCount: contentLines.length,
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