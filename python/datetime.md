```
import datetime
data1=datetime.date.today()
print(date1)

year=datetime.date.today().year
print(year)

month=datetime.date.today().month
print(month)

date1=datetime.date(year=year,month=month,day=1)
print(date1)

date2=datetime.date(year=year,month=month,day=2)
print(date2)

import caclendar
x,y=calendar.monthrange(year,month)

```