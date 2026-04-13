import sys
import json
from predict_acads import predict_academic_stress

try:
    input_data = json.loads(sys.argv[1])
    result = predict_academic_stress(input_data)

    # ✅ ONLY THIS LINE
    print(json.dumps(result))

except Exception as e:
    print(json.dumps({"error": str(e)}))