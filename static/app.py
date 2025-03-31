import pickle
from flask import Flask, render_template, request,session,flash
import numpy as np
import os
import get
app = Flask(__name__)

import mysql.connector


conn=mysql.connector.connect(host="localhost",user="root",password="0786",autocommit=True)
mycursor=conn.cursor(dictionary=True,buffered=True)
mycursor.execute("create database if not exists faceemotion")
mycursor.execute("use faceemotion")
mycursor.execute("create table if not exists face(id int primary key auto_increment,cname varchar(255),email varchar(30) unique,cpassword text)")




app = Flask(__name__)
app.secret_key = 'super secret key'


@app.route('/')
def index ():
    return render_template('index.html')




@app.route('/registration',methods =['GET', 'POST'])
def registration():
  if request.method == 'POST' and 'pass' in request.form and 'email' in request.form and 'hos' in request.form:
        name = request.form.get('pass')
        password=request.form.get('hos')
        mob = request.form.get('mob')
        email = request.form.get('email')
        mycursor.execute("SELECT * FROM face WHERE email = '"+ email +"' ")
        account = mycursor.fetchone()
        if account:
            flash('You are already registered, please log in')
        else:
            
            mycursor.execute("insert into water values(NULL,'"+ name +"','"+ email +"','"+ password +"')")
            # msg=flash('You have successfully registered !')
            return render_template("login.html")
        
  return render_template("register.html")

@app.route('/login',methods =['GET', 'POST'])
def login():
    if request.method == 'POST' and 'nm' in request.form and 'pass' in request.form:
        print('hello')
        email = request.form['nm']
        password = request.form['pass']
        
        mycursor.execute("SELECT * FROM face WHERE email = '"+ email +"' AND cpassword = '"+ password +"'")
        account = mycursor.fetchone()
        print(account)
        if account:
            session['loggedin'] = True
            session['email'] = account['email']
            msg = flash('Logged in successfully !')
                
            return render_template('predict.html')
        else:
            msg = flash('Incorrect username / password !')
            return render_template('login.html',msg=msg)
    return render_template('login.html')





# run application
if __name__ == "__main__":
    app.run(debug=True)
