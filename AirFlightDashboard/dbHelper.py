import mysql.connector

class DB:
    def __init__(self):
        try:
            self.conn = mysql.connector.connect(host="127.0.0.1",user="root",password="",
                                                database="assignment")
            self.cur = self.conn.cursor()
        except:
            print("Connection error.")
    def fetch_city_names(self):
        city = []
        self.cur.execute("SELECT DISTINCT(Destination) FROM flights UNION SELECT DISTINCT(source) FROM flights")
        data = self.cur.fetchall()
        for item in data:
            city.append(item[0])
        return city
    
    def fetch_all_flights(self,source,destination):
        self.cur.execute("""SELECT Airline,Route,Dep_time,Duration,Price FROM flights WHERE source = '{}' AND destination = '{}' """.format(source,destination))
        data = self.cur.fetchall()
        return data
    
    def fetch_airline_frequency(self):
        airline = []
        frequency = []
        self.cur.execute("""SELECT Airline,COUNT(*) FROM flights GROUP BY airline""")
        data = self.cur.fetchall()
        for item in data:
            airline.append(item[0])
            frequency.append(item[1])
        return airline,frequency
    
    def busy_airport(self):
        city = []
        frequency = []
        self.cur.execute("""SELECT Source,COUNT(*) FROM (SELECT Source FROM Flights UNION ALL SELECT Destination 
                         FROM Flights) t GROUP BY t.Source ORDER BY COUNT(*) DESC""")
        data = self.cur.fetchall()
        for item in data:
            city.append(item[0])
            frequency.append(item[1])
        return city,frequency
    
    def daily_frequency(self):
        date = []
        frequency = []
        self.cur.execute("""SELECT Date_of_Journey,COUNT(*) FROM flights GROUP BY Date_of_Journey""")
        data = self.cur.fetchall()
        for item in data:
            date.append(item[0])
            frequency.append(item[1])
        return date,frequency 