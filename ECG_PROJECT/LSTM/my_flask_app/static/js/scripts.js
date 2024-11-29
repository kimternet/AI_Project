document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('predict-form').addEventListener('submit', function(event) {
        event.preventDefault(); // 폼 제출을 막고 AJAX 요청을 사용합니다.

        var hr = document.getElementById('hr').value;
        var resp = document.getElementById('resp').value;
        var spo2 = document.getElementById('spo2').value;
        var temp = document.getElementById('temp').value;

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `hr=${hr}&resp=${resp}&spo2=${spo2}&temp=${temp}`,
        })
        .then(response => response.json())
        .then(data => {
            var resultText = data.prediction === 1 ? 'Abnormal' : 'Normal';
            document.getElementById('result').textContent = `Prediction: ${resultText}`;
            console.log(`Prediction result: ${resultText}`);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').textContent = 'Error occurred. Please try again.';
        });
    });
});
