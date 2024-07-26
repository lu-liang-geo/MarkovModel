#! /bin/bash

for PLOT in overall_daily overall_hourly spring_daily spring_hourly summer_daily summer_hourly fall_daily fall_hourly winter_daily winter_hourly
do
	dot -Tpng "$PLOT.dot" > "$PLOT.png"
done