import sys
import json
from predict_acads import predict_academic_stress   # ⚠️ your file name here

try:
    input_data = json.loads(sys.argv[1])
    
    result = predict_academic_stress(input_data)

    print(json.dumps(result))   # ✅ THIS LINE IS EVERYTHING

except Exception as e:
    print(json.dumps({"error": str(e)}))