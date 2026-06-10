import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder="templates")

# Load models
model = pickle.load(open('model.pkl', 'rb'))      # Placement model
model1 = pickle.load(open('model1.pkl', 'rb'))    # Salary model

@app.route('/')
def h():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():

    cgpa = request.args.get('cgpa', '0')
    projects = request.args.get('projects', '0')
    workshops = request.args.get('workshops', '0')
    mini_projects = request.args.get('mini_projects', '0')
    skills = request.args.get('skills', '')
    communication_skills = request.args.get('communication_skills', '0')
    internship = request.args.get('internship', '0')
    hackathon = request.args.get('hackathon', '0')
    tw_percentage = request.args.get('tw_percentage', '0')
    te_percentage = request.args.get('te_percentage', '0')
    backlogs = request.args.get('backlogs', '0')
    name = request.args.get('name', 'Student')

    # Count number of skills entered
    if skills.strip() == "":
        s = 0
    else:
        s = len(skills.split(','))

    # Placement prediction
    arr = np.array([
        cgpa,
        projects,
        workshops,
        mini_projects,
        s,
        communication_skills,
        internship,
        hackathon,
        tw_percentage,
        te_percentage,
        backlogs
    ])

    brr = np.asarray(arr, dtype=float)

    output = model.predict([brr])[0]

    print("Placement prediction:", output)

    # Placement status for salary model
    p = 1 if output == 'Placed' else 0

    # Salary prediction
    arr1=np.array([
        cgpa,
        projects,
        workshops,
        mini_projects,
        s,
        communication_skills,
        internship,
        hackathon,
        tw_percentage,
        te_percentage,
        backlogs,
        p
    ])

    brr1=np.asarray(arr1,dtype=float)

    print("Salary features:")
    print(brr1)

    salary=model1.predict([brr1])

    print("Predicted salary:", salary)
    salary_value = int(salary[0])

    # Format salary with commas
    formatted_salary = f"{salary_value:,}"

    if output == 'Placed':
        out = f'Congratulations {name} !! You have high chances of getting placed!!!'
        out2 = f'Your Expected Salary will be INR {formatted_salary} per annum'
    else:
        out = f'Sorry {name} !! You have low chances of getting placed. All the best!!!!'
        out2 = 'Improve your skills...'

    return render_template(
        'output.html',
        output=out,
        output2=out2
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )