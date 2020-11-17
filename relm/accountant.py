class PrivacyAccountant:
    def __init__(self, privacy_budget):
        self.privacy_budget = privacy_budget
        self.mechanisms = []

    @property
    def privacy_consumed(self):
        return sum(m.get_privacy_consumption() for m in self.mechanisms)

    def check_valid(self):
        return self.privacy_consumed < self.privacy_budget

    def add_mechanism(self, mechanism):
        self.mechanisms.append(mechanism)
