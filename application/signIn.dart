import 'package:flutter/material.dart';
import './filesConetent.dart';

import 'package:google_sign_in/google_sign_in.dart';
import 'package:googleapis/drive/v3.dart' as drive;
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:path/path.dart' as path;

class BackupScreen extends StatefulWidget {
  @override
  _BackupScreenState createState() => _BackupScreenState();
}

class _BackupScreenState extends State<BackupScreen> {
  GoogleSignInAccount? _user;
  List<drive.File> _files = [];
  List<drive.File> txtFiles = [];
  List<drive.File> mp4Files = [];
  bool _loading = false; // 추가: 로딩 상태를 저장하는 변수

  Future<void> _signIn() async {
    await BackUpRepository().signOut();
    GoogleSignInAccount? user = await BackUpRepository().signIn();
    setState(() {
      _user = user;
    });

    if (_user != null) {
      _getFiles(); // 로그인 성공 후 파일 목록을 불러옵니다.
    }
  }

  Future<void> _signOut() async {
    await BackUpRepository().signOut();
    setState(() {
      _user = null;
      _files.clear(); // 로그아웃 시 파일 목록도 초기화
      txtFiles.clear(); // 로그아웃 시 텍스트 파일 목록도 초기화
      mp4Files.clear();
    });
  }

  Future<void> _getFiles() async {
    if (_user != null) {
      _loading = true;
      _files.clear();
      txtFiles.clear();

      // 사용자의 구글 드라이브 API 가져오기
      drive.DriveApi? driveApi = await BackUpRepository().getDriveApi(_user!);
      if (driveApi != null) {
        try {
          // 파일 목록 가져오기
          drive.FileList fileList = await driveApi.files.list();

          // 가져온 파일 목록을 _files에 할당
          setState(() {
            _files = fileList.files!;
          });

          setState(() {
            // .txt 파일 필터링
            txtFiles = _files.where((file) {
              return file.mimeType != 'application/vnd.google-apps.folder' &&
                  file.name!.toLowerCase().endsWith('.txt');
            }).toList();

            // .mp4 파일 필터링 (해당 .txt 파일과 같은 이름이 있는 경우)
            txtFiles.removeWhere((txtFile) {
              String mp4FileName = txtFile.name!.replaceAll('.txt', '.mp4');

              bool mp4FileExists = _files.any((file) =>
              file.name == mp4FileName &&
                  file.mimeType != 'application/vnd.google-apps.folder' &&
                  file.name!.toLowerCase().endsWith('.mp4'));

              return !mp4FileExists;
            });
            txtFiles.sort((a, b) => a.name!.compareTo(b.name!));
            txtFiles = txtFiles.map((file) {
              return drive.File.fromJson(file.toJson())
                ..name = path.basenameWithoutExtension(file.name!);
            }).toList();
          });
        } catch (error) {
          print('파일 목록을 가져오는 중 오류가 발생했습니다: $error');
        }
      }
      _loading = false;
    }
  }

  Future<void> _getFileContent(drive.File file) async {
    if (_user != null) {
      drive.DriveApi? driveApi = await BackUpRepository().getDriveApi(_user!);
      if (driveApi != null) {
        try {
          // mp4 파일 확인 부분
          String mp4FileName = file.name! + '.mp4';

          drive.FileList mp4FileList = await driveApi.files.list(
              q: "name='$mp4FileName'");

          if (mp4FileList.files!.isNotEmpty) {
            drive.File mp4File = mp4FileList.files![0];

            String mp4FileId = mp4File.id!;
            String thumbnailUrl = 'https://drive.google.com/thumbnail?id=$mp4FileId';

            // print('영상주소: $mp4FileId');

            // txt 파일 내용 가져오기 부분
            drive.Media media = await driveApi.files.get(
              file.id!,
              downloadOptions: drive.DownloadOptions.fullMedia,
            ) as drive.Media;

            List<int> byteList = [];
            await media.stream.forEach((chunk) => byteList.addAll(chunk));
            String content = utf8.decode(byteList);

            List<String> items = content.split(',');

            String formattedTextContent = items.map((item) => item.trim()).join(
                '\n');

            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) =>
                    FileContentScreen(
                      content: formattedTextContent,
                      fileName: file.name!,
                      mp4FileName: mp4FileName,
                      mp4FileId: mp4FileId,
                      thumbnailUrl: thumbnailUrl,
                    ),
              ),
            );
          } else {
            print('해당하는 mp4 파일이 없습니다.');
          }
        } catch (error) {
          print('파일 내용을 가져오는 중 오류가 발생했습니다: $error');
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Color(0xFFF4F7FF),
        elevation: 0,
        actions: <Widget>[
          _user != null
              ? Row(
            children: [
              GestureDetector(
                onTap: _signOut,
                child: Row(
                  children: [
                    //  SizedBox(width: 8),  // 아이콘과 텍스트 사이의 간격을 조절하세요.
                    Text(
                      '로그아웃',
                      style: TextStyle(
                        fontSize: 17.0,
                        color: Colors.black,  // 텍스트 색상을 검정색으로 설정
                      ),
                    ),
                    SizedBox(width: 10),  // 텍스트와 아이콘 사이의 간격을 조절하세요.
                    Image.asset(
                      'assets/logout.png',  // 'logout.png' 이미지 경로에 따라 수정하세요.
                      height: 30,  // 아이콘의 높이 설정
                      width: 30,   // 아이콘의 너비 설정
                    ),
                    SizedBox(width: 20),
                  ],
                ),
              ),
            ],
          )
              : SizedBox(),
        ],
      ),
      body: Center(
        child: _user == null
            ? Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.asset(
              'assets/logo.png',
              height: 150.0,
            ),
            SizedBox(height: 20.0),
            Text(
              '신호위반 제보 시스템',
              style: TextStyle(fontSize: 24.0, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 100.0),
            ElevatedButton(
              onPressed: _signIn,
              style: ElevatedButton.styleFrom(
                shape: RoundedRectangleBorder(
                  borderRadius:
                  BorderRadius.circular(10), // 원하는 정도의 둥글기로 조절
                ),
                backgroundColor: Color(0x5463d6),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Image.asset(
                    'assets/google.png',
                    height: 70.0,
                  ),
                  SizedBox(width: 20.0),
                  Text(
                    '구글 계정으로 로그인',
                    style: TextStyle(color: Colors.white),
                  ),
                  SizedBox(width: 10.0),
                ],

              ),
            ),
          ],
        )
            : Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () async {
                setState(() {
                  _loading = true;
                });

                await _getFiles();

                setState(() {
                  _loading = false;
                });
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Color(0xFFF4F7FF), // 배경색을 lightBlue[50]으로 설정
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(40),
                  side: BorderSide(color: Colors.transparent),
                ),
                elevation: 0,
              ),
              child: Stack(
                alignment: Alignment.center,
                children: [
                  if (_loading)
                    Column(
                      children: [
                        Image.asset(
                          'assets/loading.png', // refresh 이미지 경로로 수정
                          height: 90.0,
                        ),
                        SizedBox(height: 20),
                        Text(
                          '불러오는 중',
                          style: TextStyle(
                            fontSize: 18.0,
                            color: Colors.black,  // 텍스트 색상을 검정색으로 설정
                          ),
                        ),
                      ],
                    ),

                  if (!_loading)
                    Row(
                      children: [
                        Padding(
                          padding: EdgeInsets.only(left: 20.0, right: 50.0),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                '영상 목록',
                                style: TextStyle(
                                  fontSize: 35.0,
                                  color: Colors.black,
                                ),
                              ),
                            ],
                          ),
                        ),
                        SizedBox(width: 80.0),
                        Column(
                          children: [
                            Image.asset(
                              'assets/refresh.png', // refresh 이미지 경로로 수정
                              height: 30.0,
                            ),
                            SizedBox(height: 10.0),
                            Text(
                              '새로고침',
                              style: TextStyle(
                                fontSize: 14.0,
                                color: Colors.black, // 텍스트 색상을 검정색으로 설정
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  SizedBox(height: 150),
                ],
              ),
            ),

            if (txtFiles.isNotEmpty)
              Expanded(
                  child: ListView.separated(
                    itemCount: txtFiles.length,
                    separatorBuilder: (BuildContext context, int index) => Divider(),
                    itemBuilder: (BuildContext context, int index) {
                      return Container(
                        color: Color(0xFFF4F7FF),
                        child: Column(
                          children: [
                            ListTile(
                              contentPadding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 10.0),
                              leading: Padding(
                                padding: EdgeInsets.only(left: 24.0),
                                child: Image.asset('assets/video.png', width: 40, height: 40),
                              ),
                              title: Padding(
                                padding: EdgeInsets.only(left: 8.0),
                                child: Text(
                                  txtFiles[index].name!,
                                  style: TextStyle(
                                    fontSize: 19.0,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                              onTap: () {
                                _getFileContent(txtFiles[index]);
                              },
                            ),
                            Divider( // 리스트 항목 사이에 Divider를 추가합니다.
                              thickness: 1, // 선의 두께를 설정합니다.
                            ),
                          ],
                        ),
                      );
                    },
                  )
              ),
          ],
        ),
      ),
    );
  }
}

class BackUpRepository {
  Future<GoogleSignInAccount?> signIn() async {
    GoogleSignIn googleSignIn = GoogleSignIn(scopes: [
      drive.DriveApi.driveAppdataScope,
      drive.DriveApi.driveReadonlyScope,
    ]);

    return await googleSignIn.signInSilently() ?? await googleSignIn.signIn();
  }

  Future<void> signOut() async {
    GoogleSignIn googleSignIn = GoogleSignIn();
    await googleSignIn.signOut();
  }

  Future<drive.DriveApi?> getDriveApi(
      GoogleSignInAccount googleSignInAccount) async {
    final header = await googleSignInAccount.authHeaders;
    GoogleAuthClient googleAuthClient = GoogleAuthClient(header: header);
    return drive.DriveApi(googleAuthClient);
  }
}

class GoogleAuthClient extends http.BaseClient {
  final Map<String, String> header;
  final http.Client client = http.Client();

  GoogleAuthClient({required this.header});

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) {
    request.headers.addAll(header);
    return client.send(request);
  }
}