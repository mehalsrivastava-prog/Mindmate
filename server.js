const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");

const app = express();

app.use(cors());
app.use(express.json());

// ─── Predict Route ─────────────────────────────
app.post("/predict", (req, res) => {
  const inputData = req.body;

  // Run Python script
  const py = spawn("python", ["predict.py", JSON.stringify(inputData)]);

  let result = "";
  let error = "";

  py.stdout.on("data", (data) => {
    result += data.toString();
  });

  py.stderr.on("data", (data) => {
    error += data.toString();
  });

  py.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: error });
    }

    try {
      const output = JSON.parse(result);
      res.json(output);
    } catch (err) {
      res.status(500).json({ error: "Invalid response from model" });
    }
  });
});

// ─── Start Server ─────────────────────────────
app.listen(3000, () => {
  console.log("🚀 Server running on http://localhost:3000");
});