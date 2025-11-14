# Flight Fare Prediction (Flask)

Project that serves a machine-learning model via a small Flask web app to predict flight fares based on user form input.

**Repository layout**

- `app.py` : Flask application that exposes a form and `/predict` endpoint.
- `c1_flight_rf_new.pkl` : Pickled trained model (must be present in project root for the app to work).
- `requirements.txt` : Python dependencies for the project.
- `myvenv/` : (Optional) virtual environment included in this workspace.
- `templates/home.html` : Frontend form used to gather inputs from user.
- `static/css/styles.css` : Styling for `home.html`.

## Project overview

This Flask application loads a pre-trained model (`c1_flight_rf_new.pkl`) and exposes a web form (home page) that allows a user to enter flight details (departure/arrival times, airline, stops, source and destination). When the form is submitted the app extracts and preprocesses features, calls the model to predict a fare, and returns the predicted fare on the same web page.

## Main endpoints

- `GET /` : Renders `templates/home.html` (the input form).
- `POST /predict` : Accepts form inputs, computes features, and returns a predicted flight price.

## Expected form fields (keys used by `app.py`)

- `Dep_Time` : departure date/time (HTML datetime-local), example format produced by form: `YYYY-MM-DDTHH:MM`.
- `Arrival_Time` : arrival date/time (HTML datetime-local), same format.
- `stops` : integer number of stops (e.g., 0,1,2).
- `airline` : airline name string; code expects values such as `Jet Airways`, `IndiGo`, `Air India`, `Multiple carriers`, `SpiceJet`, `Vistara`, `GoAir`, or other (falls back to `Other`).
- `Source` : departure city string — expected values include `Delhi`, `Kolkata`, `Mumbai`, `Chennai` (others treated as none).
- `Destination` : arrival city string — expected values include `Cochin`, `Delhi`, `Hyderabad`, `Kolkata` (others treated as none).

## How input is processed (feature extraction)

The app calculates the following numeric features before calling `model.predict` (and passes them in the following order):

1. `Total_Stops` (integer)
2. `journey_day` (int) — day of month from `Dep_Time`
3. `journey_month` (int) — month from `Dep_Time`
4. `dep_hour` (int) — hour of departure
5. `dep_min` (int) — minute of departure
6. `arrival_hour` (int) — hour of arrival
7. `arrival_min` (int) — minute of arrival
8. `Duration_hour` (int) — absolute difference between arrival and departure hours
9. `Duration_mins` (int) — absolute difference between arrival and departure minutes
10. One-hot airline columns in this order: `Airline_AirIndia`, `Airline_GoAir`, `Airline_IndiGo`, `Airline_JetAirways`, `Airline_MultipleCarriers`, `Airline_Other`, `Airline_SpiceJet`, `Airline_Vistara`
11. One-hot source columns: `Source_Chennai`, `Source_Kolkata`, `Source_Mumbai` (note: `Source_Delhi` is treated separately in code but not included directly in the explicit features list passed to the model)
12. One-hot destination columns: `Destination_Cochin`, `Destination_Delhi`, `Destination_Hyderabad`, `Destination_Kolkata`

Note: The app uses `pandas.to_datetime` to parse the `datetime-local` style string and extracts day/month/hour/minute values.

## Model

- The model file expected by the app is `c1_flight_rf_new.pkl` in the project root. The code calls `pickle.load(open("c1_flight_rf_new.pkl", "rb"))` on startup.
- The app sends a single sample row (list of features) to `model.predict` and rounds the resulting prediction to two decimal places.

## Setup and run (Windows PowerShell)

1. (Optional) If you want to create and use a virtual environment yourself (recommended):

```powershell
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Ensure `c1_flight_rf_new.pkl` is present in the project root.

4. Run the Flask app:

```powershell
python app.py
```

5. Open the app in a browser at `http://127.0.0.1:5000` and use the form to get predictions.

## Example: test `POST /predict` with curl (form data)

Replace values below with suitable examples; this demonstrates how to call the endpoint directly.

```powershell
curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/x-www-form-urlencoded" -d "Dep_Time=2025-11-14T09:30&Arrival_Time=2025-11-14T12:15&stops=1&airline=IndiGo&Source=Delhi&Destination=Cochin"
```

The server returns the rendered `home.html` containing the `prediction_text` with the predicted fare.

## Troubleshooting

- If Flask fails to start because the model file is missing, confirm `c1_flight_rf_new.pkl` is in the same directory as `app.py`.
- If you see parse errors for date/time values, ensure the form uses an HTML `datetime-local` input or send `YYYY-MM-DDTHH:MM` style strings when posting manually.
- If module import errors occur, check your virtual environment and that `pip install -r requirements.txt` completed successfully.

## Notes and next steps

- The feature engineering is minimal (absolute hour/min differences); you may want to verify duration calculation logic for overnight flights or negative differences.
- Consider adding input validation and clearer handling of unexpected airline/source/destination values.

---

If you want, I can also:
- Add a simple README badge and usage screenshots.
- Add a small `run.sh`/`run.ps1` helper to activate the venv and start the app.
- Add unit tests for the input parsing and prediction route.
