import random
from itertools import cycle
from collections import deque

from core.controllers.traffic_signal_controller import TrafficSignalController

class MaxPressureTSC(TrafficSignalController):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, green_t):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t)
        self.green_t = green_t
        self.t = 0
        self.phase_deque = deque()
        self.max_pressure_lanes = self.max_pressure_lanes()
        self.data = None
        self.phase_g_count = {}
        for p in self.green_phases:
            self.phase_g_count[p] = sum([1 for m in p if m == 'g' or m == 'G'])

    def next_phase(self):
        if len(self.phase_deque) == 0:
            max_pressure_phase = self.max_pressure()
            phases = self.get_intermediate_phases(self.phase, max_pressure_phase)
            self.phase_deque.extend(phases+[max_pressure_phase])
        return self.phase_deque.popleft()

    def max_pressure_lanes(self):

        max_pressure_lanes = {}
        for g in self.green_phases:
            inc_lanes = set()
            out_lanes = set()
            for l in self.phase_lanes[g]:
                inc_lanes.add(l)
                for ol in self.netdata['lane'][l]['outgoing']:
                    out_lanes.add(ol)

            max_pressure_lanes[g] = {'inc':inc_lanes, 'out':out_lanes}
        return max_pressure_lanes

    def max_pressure(self):
        phase_pressure = {}
        no_vehicle_phases = []
        for g in self.green_phases:
            inc_lanes = self.max_pressure_lanes[g]['inc']
            out_lanes = self.max_pressure_lanes[g]['out']
            inc_pressure = sum([ len(self.data[l]) if l in self.data else 0 for l in inc_lanes])
            out_pressure = sum([ len(self.data[l]) if l in self.data else 0 for l in out_lanes])
            phase_pressure[g] = inc_pressure - out_pressure
            if inc_pressure == 0 and out_pressure == 0:
                no_vehicle_phases.append(g)

        if len(no_vehicle_phases) == len(self.green_phases):
            return random.choice(self.green_phases)
        else:
            phase_pressure = [ (p, phase_pressure[p]) for p in phase_pressure]
            phase_pressure = sorted(phase_pressure, key=lambda p:p[1], reverse=True)
            phase_pressure = [ p for p in phase_pressure if p[1] == phase_pressure[0][1] ]
            return random.choice(phase_pressure)[0]

    def next_phase_duration(self):
        if self.phase in self.green_phases:
            return self.green_t
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    def update(self, data):
        self.data = data 
