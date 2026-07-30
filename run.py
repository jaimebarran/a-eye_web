from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
    
# Finish open port
# lsof -i :5000
# kill -9 <process_id>
