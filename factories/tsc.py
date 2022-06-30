from core.controllers.max_pressure import MaxPressureTSC
from core.controllers.next_phase_rl import NextPhaseRLTSC
from core.controllers.next_duration_rl import NextDurationRLTSC
from core.controllers.websters import WebstersTSC
from core.controllers.fixed import FixedCycleTSC
from factories.rl import rl_factory


def tsc_factory(tsc_type, tl, args, netdata, rl_stats, exp_replay, neural_network, eps, conn):
    if tsc_type == 'maxpressure':
        return MaxPressureTSC(conn, tl, args.mode, netdata, args.r, args.y,
                              args.g_min )
    elif tsc_type == 'websters':
        return WebstersTSC(conn, tl, args.mode, netdata, args.r, args.y, args.g_min, 
                        args.c_min, args.c_max, args.sat_flow, args.update_freq)
    elif tsc_type == 'fixed':
        return FixedCycleTSC(conn, tl, args.mode, netdata, args.r, args.y, args.g_min)
    elif tsc_type == 'dqn':
        dqnagent = rl_factory(tsc_type, args,
                              neural_network, exp_replay, rl_stats, len(netdata['inter'][tl]['green_phases']), eps)
        return NextPhaseRLTSC(conn, tl, args.mode, netdata, args.r, args.y,
                              args.g_min, dqnagent)
    elif tsc_type == 'ddpg':
        ddpgagent = rl_factory(tsc_type, args,
                                neural_network, exp_replay, rl_stats, 1, eps)
        return NextDurationRLTSC(conn, tl, args.mode, netdata, args.r, args.y,
                                 args.g_min, args.g_max, ddpgagent)
    else:
        assert 0, 'Supplied traffic signal control argument type '+str(tsc_type)+' does not exist.'
