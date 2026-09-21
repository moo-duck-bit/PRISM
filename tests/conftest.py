"""저장소 루트를 경로에 넣어 prism 패키지를 임포트할 수 있게 한다."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
