# Regression-Prediction-Based-on-Lodon-Bike-Sharing-Dataset
## This is mainly a project for pattern recongnition course based on London Bike Sharing Dataset.Main purpose of the project is to predict the future bike shares. Therefore, two types od regression method are introduced in this project to do the regression work. One is Ridge regression and the other is Random Forest Regression.<br>
data description：<br>
"timestamp" - timestamp field for grouping the data <br>
"cnt" - the count of a new bike shares<br>
"t1" - real temperature in C<br>
"t2" - temperature in C "feels like"<br>
"hum" - humidity in percentage <br>
"wind_speed" - wind speed in km/h<br>
"weather_code" - category of the weather<br>
"is_holiday" - boolean field - 1 holiday / 0 non holiday<br> 
"is_weekend" - boolean field - 1 if the day is weekend <br>
"season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.<br>

"weathe_code" category description:<br> 
1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity<br> 
2 = scattered clouds / few clouds<br> 
3 = Broken clouds<br> 
4 = Cloudy<br>
7 = Rain/ light Rain shower/ Light rain<br>
10 = rain with thunderstorm<br> 
26 = snowfall<br> 
94 = Freezing Fog<br>
