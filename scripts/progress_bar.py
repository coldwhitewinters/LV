#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import progressbar
import time

class ProgressBar:
    def __init__(self, max_value):
        time.sleep(0.5)
        self.bar = progressbar.ProgressBar(
            min_value=0,
            max_value=max_value,
            initial_value=0,
            widgets = [progressbar.SimpleProgress(), 
                       progressbar.Bar(), 
                       progressbar.Percentage()])
        self.bar.update(0)
        self.counter = 0
    
    def update(self):
        self.bar.update(self.counter + 1)
        self.counter += 1
        
    def finish(self):
        self.bar.finish()
        