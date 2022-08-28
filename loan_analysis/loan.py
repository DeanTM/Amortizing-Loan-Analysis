import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class AmortizingLoan:
    """A class that keeps track of the repayment of loans."""
    def __init__(self, initial_principal, full_term):
        self.initial_principal = initial_principal
        self.full_term = full_term
        self.reset()

    def reset(self):
        self.remaining_principals = [self.initial_principal]
        self.interests_paid = []
        self.principal_reductions = []
        self.payments_made = []
        self.remaining_term = self.full_term
    
    def split_repayment(self, payment, interest_rate_annual):
        remaining_principal = self.remaining_principals[-1]
        interest = remaining_principal * interest_rate_annual/12
        principal_reduction = payment - interest
        remaining_principal_new = remaining_principal - principal_reduction
        return interest, principal_reduction, remaining_principal_new

    def payment_update(self, payment, interest_rate_annual):
        interest, principal_reduction,\
        remaining_principal_new = self.split_repayment(
            payment, interest_rate_annual)
        self.interests_paid.append(interest)
        self.principal_reductions.append(principal_reduction)
        self.remaining_principals.append(remaining_principal_new)
        self.payments_made.append(payment)
        self.remaining_term -= 1
        return None

    def payment_update_collection(self, payments, interest_rates):
        for payment,interest_rate in zip(payments, interest_rates):
            self.payment_update(
                payment=payment,
                interest_rate_annual=interest_rate)
        return None

    def pay_interest_rates(self, interest_rates):
        for ir in interest_rates:
            payment = self.get_amortized_payment_amount(
                interest_rate_annual=ir)
            self.payment_update(
                payment=payment,
                interest_rate_annual=ir)
        return None


    def get_amortized_payment_amount(
        self,
        interest_rate_annual,
        ):
        # from here:
        # https://www.educba.com/amortized-loan-formula/
        r = interest_rate_annual
        n = 12
        num_payments_left = self.remaining_term
        t_times_n = num_payments_left
        P = self.remaining_principals[-1]
        return P * (r/n) * (1+r/n)**t_times_n /((1+r/n)**t_times_n -1)

    def get_data(self):
        data_dict = dict(
            payment_number=np.arange(len(self.payments_made)),
            principal_before=self.remaining_principals[:-1],
            principal_after=self.remaining_principals[1:],
            interest_paid=self.interests_paid,
            principal_paid=self.principal_reductions,
            total_paid=self.payments_made)
        return pd.DataFrame.from_dict(data_dict).set_index('payment_number')


if __name__ == '__main__':
    ## example code
    principal = 250000.
    total_payemts = 480

    loan_35anos_variable = AmortizingLoan(
        initial_principal=principal,
        full_term=total_payemts)
    payment = 717.10
    interest_rate = 0.01566
    for i in range(total_payemts):
        loan_35anos_variable.payment_update(
            payment=payment,
            interest_rate_annual=interest_rate)



    fig = plt.figure(figsize=(8,3))
    plt.plot(
        np.cumsum(loan_35anos_variable.interests_paid),
        label='cumulative interest paid',
        color="C0")
    plt.plot(
        np.cumsum(loan_35anos_variable.principal_reductions),
        label='cumulative principal paid',
        color="C0", ls="--")
    plt.axhline(principal, ls=":", color='black')
    plt.xlabel("Month")
    plt.ylabel("Cumulative Payment (euros)")
    plt.legend()
    plt.show()
