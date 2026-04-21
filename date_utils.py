from datetime import date, timedelta





class DayCount:
    def __init__(self, convention):
        self.convention = convention


    def year_fraction(self, date1, date2):
        if date1 < date2:
            if self.convention == "ACT/365":
                return (date2 - date1).days / 365

            if self.convention == "ACT/360":
                return (date2 - date1).days / 360


            if self.convention == "30/360":
                Y1, M1, D1 = date1.year, date1.month, date1.day
                Y2, M2, D2 = date2.year, date2.month, date2.day
                if D1 == 31:
                    D1 = 30
                if D2 == 31 and D1 == 30:
                    D2 = 30
                return (360*(Y2-Y1) + 30*(M2-M1) + (D2-D1))/360

        elif date1 == date2:
            return 0

        raise ValueError("Date de début doit être inférieure à la date de fin")


class ScheduleGenerator:
    def __init__(self, issue_date, maturity_date, freq):
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.freq = freq
    def freq_to_months(self):
        if self.freq > 0 and 12 % self.freq == 0:
            freq_in_month = int(12/self.freq)
            return freq_in_month
        raise ValueError("Frequence doit être correcte et positive")


    def generate_dates(self):
        payment_dates = []
        current_date = self.issue_date
        frq = self.freq_to_months()
        if self.issue_date > self.maturity_date:
            raise ValueError("Issue date must be before maturity date")
        while current_date < self.maturity_date:
            Y = current_date.year
            M = current_date.month - 1
            D = current_date.day
            M_new = M + frq
            annee_restant = M_new // 12
            M_new = M_new % 12
            M_new +=1
            Y += annee_restant
            try:
                new_date = date(Y, M_new, D)
            except ValueError:
                if M_new < 12:
                    date_to_find_last_day = date(Y, M_new+1, 1)
                    last_date_month = date_to_find_last_day - timedelta(days=1)
                    new_date = last_date_month
                else:
                    date_to_find_last_day = date(Y+1, 1, 1)
                    last_date_month = date_to_find_last_day - timedelta(days=1)
                    new_date = last_date_month
            if new_date <= self.maturity_date:
                payment_dates.append(new_date)
                current_date = new_date
            else:
                break

        return payment_dates




