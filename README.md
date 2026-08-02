# claude-study

파이썬과 pandas를 배우기 위한 개인 학습 프로젝트로, 가상의 매출 데이터를 만들어
지역별/월별/제품별로 분석하고 표와 그래프로 보여준다.

## 폴더 구조

```
claude-study/
├── data/
│   ├── make_data.py      # 연습용 매출 데이터(sales.csv)를 생성하는 스크립트
│   ├── sales.csv         # 분석에 사용하는 정상 매출 데이터
│   └── sales_dirty.csv   # 결측치/음수/공백 등 이상값이 섞인 테스트용 데이터
├── src/                  # 분석 로직 (다른 파일에서 가져다 쓰는 함수 모음)
│   ├── data.py           # 데이터 전처리 함수 (날짜 -> 연월/분기 컬럼 추가 등)
│   └── metrics.py        # 지역별/월별/제품별 매출 집계 함수
├── images/               # report.ipynb를 실행하면 그래프 이미지가 저장되는 폴더
├── reports/              # report.ipynb를 실행하면 집계 표(csv)가 저장되는 폴더
├── report.ipynb          # 매출 리포트 노트북 (결과를 표/그래프로 보여주는 용도)
├── test.ipynb            # 학습 중 이것저것 실험해보는 스크래치 노트북 (신경 쓰지 않아도 됨)
├── data_check.py         # 데이터 품질(결측치/음수/공백) 검사 스크립트
├── test_data_check.py    # data_check.py에 대한 pytest 테스트
├── test_report.py        # src/ 함수들에 대한 pytest 테스트
├── basics_01_data.py     # 파이썬 기초 문법 연습 (변수, 리스트, 딕셔너리 등)
├── basics_02_flow.py     # 파이썬 기초 문법 연습 (조건문, 반복문, 함수)
├── hello.py               # 가장 간단한 입출력 연습 파일
├── requirements.md       # 이 프로젝트가 풀고자 하는 분석 요구사항 문서
└── requirements.txt      # pip freeze로 만든 라이브러리 버전 목록
```

## 설치 방법 (Windows, PowerShell 기준)

1. 이 저장소를 내려받는다 (git clone 또는 zip 다운로드).
2. PowerShell에서 프로젝트 폴더로 이동한다.
   ```powershell
   cd C:\경로\claude-study
   ```
3. 가상환경을 만든다. (`.venv`라는 이름의 폴더가 새로 생긴다)
   ```powershell
   python -m venv .venv
   ```
4. 가상환경을 활성화한다.
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   - 만약 "이 시스템에서 스크립트를 실행할 수 없습니다" 같은 오류가 나면, PowerShell 실행 정책 때문이다.
     아래 명령을 한 번 실행한 뒤 다시 시도한다.
     ```powershell
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
     ```
   - 활성화에 성공하면 프롬프트 맨 앞에 `(.venv)`가 표시된다.
5. 필요한 라이브러리를 설치한다.
   ```powershell
   pip install -r requirements.txt
   ```

## 실행 방법

**아래 명령들은 모두 가상환경을 활성화한 상태(`(.venv)`가 보이는 상태)에서 실행한다.**

### report.ipynb 열어서 실행하기

- VS Code를 쓴다면: VS Code에서 이 폴더를 열고 `report.ipynb`를 클릭한 뒤,
  오른쪽 위에서 커널로 `.venv`를 선택한다. 그다음 상단 메뉴에서
  **"Restart & Run All"**(전체 재실행)을 누르면 처음부터 끝까지 실행된다.
- 또는 터미널에서 Jupyter를 직접 실행할 수도 있다.
  ```powershell
  jupyter lab
  ```
  브라우저가 열리면 `report.ipynb`를 더블클릭해서 연 뒤, 상단 메뉴에서
  **Run > Run All Cells**를 누른다.
- 정상적으로 실행되면 `images/` 폴더에 그래프 이미지가, `reports/` 폴더에
  집계 결과 csv 파일이 새로 저장(덮어쓰기)된다.

### 테스트(pytest) 실행하기

프로젝트 루트 폴더에서 아래 명령을 실행한다.

```powershell
pytest
```

`src/`와 `data_check.py`의 함수들이 예상대로 계산/검사하는지 확인하는
테스트이며, 모두 통과(`passed`)하면 정상이다.

### 참고: test.ipynb

`test.ipynb`는 학습 중에 이것저것 실험해보는 스크래치용 노트북이다.
정식 결과물이 아니므로 신경 쓰지 않아도 된다.
