"""Independent manufactured examples for comparison metrics and auditing."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comparison.metrics import (force_statistics, sustained_entry_time,
                                torque_statistics, tracking_entry_metrics,
                                vector_statistics)
from validate_comparison import (independent_common_bounds, independent_entry,
                                independent_log_transform, independent_reference,
                                select_trials, audit_control_formulas,
                                audit_external_formulas, EXTERNAL_FIELDS,
                                audit_prefix, reference_sensitivity, ComparisonAudit, HERE)


class ComparisonMetricTests(unittest.TestCase):
    def test_entry_requires_final_suffix_not_early_successful_island(self):
        times = np.arange(0., 7.)
        self.assertEqual(sustained_entry_time(times, [True, True, True, False, True, True, True]), 4.)
        self.assertIsNone(sustained_entry_time(times, [True, True, True, True, True, True, False]))

    def test_entry_rejects_short_or_terminated_tail(self):
        times = np.arange(0., 7.)
        self.assertIsNone(sustained_entry_time(times, [False]*5+[True]*2))
        self.assertIsNone(sustained_entry_time(times, [True]*7, completed=False))
        self.assertIsNone(sustained_entry_time(times, [True]*7, requested_end_s=24.))

    def test_entry_exact_two_seconds_and_component_criteria(self):
        times = np.linspace(0., 4., 9)
        error = np.zeros((9, 6))
        error[:4, 0] = .01
        error[:2, 3] = .1
        result = tracking_entry_metrics(times, error)
        self.assertEqual(result['position_entry_s'], 2.)
        self.assertEqual(result['orientation_coordinate_entry_s'], 1.)
        self.assertEqual(result['simultaneous_entry_s'], 2.)

    def test_vector_norm_does_not_mix_force_and_moment(self):
        times = np.array([0., 1.])
        actual = np.zeros((2, 2, 6))
        actual[:, 0, :3] = [[3., 4., 0.], [0., 0., 0.]]
        actual[:, 0, 3:] = [[0., 0., 12.], [0., 0., 0.]]
        result = force_statistics(times, actual, np.zeros_like(actual))
        arm = result['arms']['left']
        self.assertAlmostEqual(arm['force_xyz_N']['error_vector_rms'], 5/np.sqrt(2))
        self.assertAlmostEqual(arm['euler_moment_xyz_Nm']['error_vector_rms'], 12/np.sqrt(2))

    def test_ac_and_bias_are_separate_and_bad_samples_are_not_dropped(self):
        actual = np.array([[1., -2., 0.], [3., -2., 0.]])
        result = vector_statistics(actual, np.zeros_like(actual))
        np.testing.assert_allclose(result['actual_mean'], [2., -2., 0.])
        np.testing.assert_allclose(result['actual_ac_rms_components'], [1., 0., 0.])
        bad = actual.copy(); bad[1, 0] = np.nan
        result = vector_statistics(bad, np.zeros_like(bad))
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['nonfinite_actual_samples'], 1)
        self.assertFalse(result['all_finite'])
        self.assertTrue(np.isnan(result['error_vector_rms']))

    def test_short_window_uses_actual_elapsed_time(self):
        times = np.arange(0., 25.)
        actual = np.broadcast_to(times[:, None, None], (25, 2, 6))
        result = force_statistics(times, actual, np.zeros_like(actual), (6., 8.))
        self.assertEqual(result['samples'], 3)
        np.testing.assert_allclose(result['arms']['left']['force_xyz_N']['actual_mean'], [7., 7., 7.])

    def test_saturation_is_retained_and_incorrect_clipping_detected(self):
        times = np.array([0., 1.])
        command = np.array([[0., 3.], [-4., 0.]])
        limits = np.array([2., 2.])
        result = torque_statistics(times, command, np.clip(command, -limits, limits), limits)
        self.assertEqual(result['saturated_tick_count'], 2)
        self.assertEqual(result['maximum_raw_limit_ratio'], 2.)
        self.assertEqual(result['expected_clipping_max_error_Nm'], 0.)
        result = torque_statistics(times, command, np.zeros_like(command), limits)
        self.assertEqual(result['expected_clipping_max_error_Nm'], 2.)


class IndependentFormulaTests(unittest.TestCase):
    @staticmethod
    def settings():
        return dict(chi=[20.]*3+[8.]*3, ell=[50.]*3+[20.]*3,
            evaluation_initial_upper=[.08,.07,.07,.22,.20,.22],
            evaluation_initial_lower=[.09,.07,.08,.25,.20,.23],
            evaluation_transition_s=1.)

    def test_common_boundary_initial_values_and_matching_after_transition(self):
        from comparison.controller import common_bounds
        from types import SimpleNamespace
        cfg = self.settings()
        for time in (0., .23, .999, 1., 3.3, 5.2, 7., 24.):
            actual = independent_common_bounds(time, cfg, True)
            target = common_bounds(time, SimpleNamespace(**cfg), True)
            for x, y in zip(actual, target):
                np.testing.assert_allclose(x, y, rtol=3e-11, atol=3e-11)
        u, l, *_ = independent_common_bounds(0., cfg)
        np.testing.assert_allclose(u, cfg['evaluation_initial_upper'])
        np.testing.assert_allclose(l, cfg['evaluation_initial_lower'])

    def test_common_boundary_first_second_derivatives_from_values(self):
        cfg, h = self.settings(), 2e-5
        for time in (.2, .7, 1.5, 3.7, 5.6, 8.):
            minus = independent_common_bounds(time-h, cfg, True)
            center = independent_common_bounds(time, cfg, True)
            plus = independent_common_bounds(time+h, cfg, True)
            for value_index, derivative_index, second_index in ((0,2,4),(1,3,5)):
                first = (plus[value_index]-minus[value_index])/(2*h)
                second = (plus[value_index]-2*center[value_index]+minus[value_index])/h**2
                np.testing.assert_allclose(center[derivative_index], first, rtol=3e-6, atol=3e-8)
                np.testing.assert_allclose(center[second_index], second, rtol=8e-5, atol=2e-5)

    def test_log_barrier_derivatives_along_nonzero_velocity_path(self):
        cfg, time, h = self.settings(), .47, 1e-6
        error = np.array([.003,-.004,.006,-.01,.02,-.025])
        velocity = np.array([.02,-.01,.015,.03,-.02,.01])
        scale = np.array([.01,.02,.03,.04,.05,.06])
        center = independent_log_transform(error, velocity,
            independent_common_bounds(time,cfg),scale)
        minus = independent_log_transform(error-h*velocity, velocity,
            independent_common_bounds(time-h,cfg),scale)
        plus = independent_log_transform(error+h*velocity, velocity,
            independent_common_bounds(time+h,cfg),scale)
        np.testing.assert_allclose(center.xidot,(plus.xi-minus.xi)/(2*h),rtol=2e-7,atol=3e-9)
        np.testing.assert_allclose(center.adot,(plus.a-minus.a)/(2*h),rtol=2e-7,atol=3e-9)
        np.testing.assert_allclose(center.bdot,(plus.b-minus.b)/(2*h),rtol=2e-7,atol=3e-9)

    def test_reference_derivatives_and_short_run_do_not_change_reference(self):
        scenario = dict(reference_profile='moderate',excitation_duration=20.)
        center, h = np.zeros(6), 1e-5
        for time in (.5,1.7,8.,18.4,19.7,22.):
            value, first, second = independent_reference(np.asarray(time),center,scenario)
            minus = independent_reference(np.asarray(time-h),center,scenario)[0]
            plus = independent_reference(np.asarray(time+h),center,scenario)[0]
            np.testing.assert_allclose(first,(plus-minus)/(2*h),rtol=1e-6,atol=2e-8)
            np.testing.assert_allclose(second,(plus-2*value+minus)/h**2,rtol=2e-4,atol=2e-5)
        hold = independent_reference(np.array([0.,4.,8.]),np.arange(6.),dict(reference_profile='hold'))
        np.testing.assert_allclose(hold[0],np.broadcast_to(np.arange(6.),(3,6)))
        np.testing.assert_array_equal(hold[1],np.zeros((3,6)))

    def test_independent_entry_agrees_with_terminal_suffix_and_censoring(self):
        times = np.arange(0.,7.)
        inside = np.array([1,1,1,0,1,1,1],dtype=bool)
        self.assertEqual(independent_entry(times,inside,2.,6.,True),4.)
        self.assertIsNone(independent_entry(times,inside,2.,24.,False))
        self.assertIsNone(independent_entry(times,inside,2.,24.,True))

    def test_registered_terminated_or_missing_trials_are_not_dropped(self):
        root = Path('/tmp/manufactured_comparison')
        protocol = dict(trials=[dict(id='ok',method='proposed',scenario='nominal',path='ok'),
                                dict(id='failed',method='linear_surface',scenario='nominal',path='failed')])
        selected = select_trials(root,protocol)
        self.assertEqual([entry[1]['id'] for entry in selected],['ok','failed'])
        self.assertEqual(len(select_trials(root,protocol,method='linear_surface')),1)

    def test_full_control_audit_accepts_declared_ablations_and_rejects_changed_torque(self):
        import json
        from dataclasses import asdict
        import mujoco
        from comparison.controller import ComparisonController, ComparisonSettings
        from design_excitation import multisine_reference
        from jas.dynamics import LinkDynamics
        from run_experiments import initialize, kinematics
        metadata = json.loads((HERE/'models/scene_metadata.json').read_text())
        model = mujoco.MjModel.from_xml_path(str(HERE/'models/dual_cr7.xml'))
        data = mujoco.MjData(model)
        kins = [kinematics(metadata,name) for name in ('left','right')]
        center = np.r_[metadata['payload_nominal']['initial_position_world_m'],np.zeros(3)]
        offset = [-.015,.01,-.012,-.03,.02,-.025]
        initialize(model,data,metadata,kins,center,offset)
        payload = metadata['payload_nominal']
        scenario = dict(reference_profile='moderate',excitation_duration=20.,nonmonotonic=False,offset=offset)
        reference = lambda t,center:multisine_reference(t,center,20.,'moderate')
        for method in ('proposed','traditional_bf','linear_surface','linear_reaching','no_history'):
            settings = ComparisonSettings(method=method,learning_integrator='explicit_euler')
            ctls = [ComparisonController(kins[a],LinkDynamics(HERE/'models/arm_nominal.xml'),center,
                    payload['mass_kg'],payload['diagonal_inertia_kg_m2'],a,settings,
                    reference=reference) for a in range(2)]
            rows = []
            for tick,time in enumerate((.11,.23)):
                q = data.qpos.copy(); q[:12] += .0001*tick*np.cos(np.arange(12))
                qd = data.qvel.copy(); qd[:12] = .003*np.sin(np.arange(12)+tick)
                commands, infos = zip(*(ctl.command(time,q[6*a:6*a+6],qd[6*a:6*a+6])
                                        for a,ctl in enumerate(ctls)))
                command = np.concatenate(commands)
                applied = np.clip(command,-np.tile(metadata['torque_limits_Nm'],2),np.tile(metadata['torque_limits_Nm'],2))
                row = dict(t=time,qpos=q,qvel=qd,theta_control=np.full((2,8),.9),
                           torque_command=command,torque_applied=applied)
                for key in ('surface','xi','regressor','momentum','remainder','A'):
                    row[key] = np.array([info[key] for info in infos])
                row['learning_drive'] = np.array([info['regressor']@info['surface'] for info in infos])
                row['known_torque_left'] = np.array([info['A'].T@(
                    (applied if tick==0 else rows[tick-1]['torque_applied'])[6*a:6*a+6]
                    +info['jacobian'].T@ctls[a].fd) for a,info in enumerate(infos)])
                row['known_torque_right'] = np.array([info['A'].T@(applied[6*a:6*a+6]
                    +info['jacobian'].T@ctls[a].fd) for a,info in enumerate(infos)])
                rows.append(row)
            arrays = {key:np.array([row[key] for row in rows]) for key in rows[0]}
            summary = dict(method=method,settings=asdict(settings),scenario=scenario)
            audit = ComparisonAudit(Path('/tmp/manufactured_comparison'))
            audit_control_formulas(audit,arrays,summary,metadata,2)
            self.assertFalse(audit.report['failures'],msg=(method,audit.report['checks']))
            arrays['torque_command'][0,0] += .1
            audit = ComparisonAudit(Path('/tmp/manufactured_comparison'))
            audit_control_formulas(audit,arrays,summary,metadata,2)
            self.assertIn('sampled_torque_formula_and_exact_declared_module_ablation',audit.report['failures'])

    def test_published_ppc_audit_rebuilds_initial_funnel_and_weighted_grasp_inverse(self):
        import json
        from dataclasses import asdict
        import mujoco
        from comparison.controller import ComparisonSettings
        from comparison.external_ppc import ExternalPPC
        from design_excitation import multisine_reference
        from run_experiments import initialize, kinematics
        metadata = json.loads((HERE/'models/scene_metadata.json').read_text())
        model = mujoco.MjModel.from_xml_path(str(HERE/'models/dual_cr7.xml'))
        data = mujoco.MjData(model)
        kins = [kinematics(metadata,name) for name in ('left','right')]
        center = np.r_[metadata['payload_nominal']['initial_position_world_m'],np.zeros(3)]
        offset = [-.015,.01,-.012,-.03,.02,-.025]
        initialize(model,data,metadata,kins,center,offset)
        settings = ComparisonSettings(method='external_ppc',learning_integrator='explicit_euler')
        scenario = dict(reference_profile='moderate',excitation_duration=20.,nonmonotonic=False,offset=offset)
        reference = lambda t,center:multisine_reference(t,center,20.,'moderate')
        # Poison dynamics placeholders ensure these test commands cannot use
        # a mass/inertia oracle while exercising the shared constructor API.
        ctls = [ExternalPPC(kins[a],object(),center,object(),object(),a,settings,
                            reference=reference) for a in range(2)]
        rows = []
        for tick,time in enumerate((0.,.01)):
            q = data.qpos.copy();q[:12] += tick*.0001*np.cos(np.arange(12))
            qd = data.qvel.copy();qd[:12] += tick*.001*np.sin(np.arange(12))
            commands,infos = zip(*(ctl.command(time,q[6*a:6*a+6],qd[6*a:6*a+6])
                                    for a,ctl in enumerate(ctls)))
            row = dict(t=time,qpos=q,qvel=qd,torque_command=np.concatenate(commands))
            for key in ('surface','xi')+EXTERNAL_FIELDS:
                row[key] = np.array([info[key] for info in infos])
            for name,key in (('native_upper','upper'),('native_lower','lower'),('native_barrier_margin','barrier_margin')):
                row[name] = np.array([info[key] for info in infos])
            rows.append(row)
        arrays = {key:np.array([row[key] for row in rows]) for key in rows[0]}
        summary = dict(method='external_ppc',settings=asdict(settings),scenario=scenario,
                       external_settings=None,controller_configurations=[ctl.configuration() for ctl in ctls])
        audit = ComparisonAudit(Path('/tmp/manufactured_external_ppc'))
        audit_external_formulas(audit,arrays,summary,metadata,2)
        self.assertFalse(audit.report['failures'],msg=audit.report['checks'])
        # The paper's experimental componentwise initialization is disclosed
        # separately from its generic norm construction, not called a failure.
        custom = dict(rho_v_initial=[2.,2.,2.,4.,4.,4.])
        ctls = [ExternalPPC(kins[a],object(),center,object(),object(),a,settings,
                           reference=reference,external_settings=custom) for a in range(2)]
        for tick,time in enumerate(arrays['t']):
            commands,infos = zip(*(ctl.command(time,arrays['qpos'][tick,6*a:6*a+6],
                                        arrays['qvel'][tick,6*a:6*a+6]) for a,ctl in enumerate(ctls)))
            arrays['torque_command'][tick] = np.concatenate(commands)
            for key in ('surface','xi')+EXTERNAL_FIELDS:
                arrays[key][tick] = np.array([info[key] for info in infos])
            for name,key in (('native_upper','upper'),('native_lower','lower'),('native_barrier_margin','barrier_margin')):
                arrays[name][tick] = np.array([info[key] for info in infos])
        summary.update(external_settings=custom,controller_configurations=[ctl.configuration() for ctl in ctls])
        audit = ComparisonAudit(Path('/tmp/manufactured_external_ppc'))
        audit_external_formulas(audit,arrays,summary,metadata,2)
        self.assertFalse(audit.report['failures'],msg=audit.report['checks'])
        self.assertEqual(audit.report['checks']['external_velocity_funnel_initialization_uses_only_t0_encoders']
                         ['paper_Step_II_a_norm_plus_slack_satisfied'],[False,False])
        arrays['ppc_palm_wrench'][0,0,0] += .1
        audit = ComparisonAudit(Path('/tmp/manufactured_external_ppc'))
        audit_external_formulas(audit,arrays,summary,metadata,2)
        self.assertIn('independent_external_published_controller_formula',audit.report['failures'])

    def test_explicit_reset_failure_retains_actual_and_expected_clocks(self):
        import tempfile
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(HERE/'models/dual_cr7.xml'))
        arrays = dict(t=np.array([0.,.0005]),qpos=np.tile(model.qpos0,(2,1)),
            qvel=np.zeros((2,model.nv)),control_valid_mask=np.ones(2,dtype=bool),
            barrier_applicable=np.ones(2,dtype=bool))
        summary = dict(control_dt_s=.0005,requested_duration_s=24.,actual_end_s=.0005,
            run_status='terminated',termination_stage='physics_guard',
            termination_reason='MuJoCo bad-state warning or automatic clock reset',failed_attempt_time_s=.001)
        with tempfile.TemporaryDirectory() as task_directory:
            directory = Path(task_directory)
            np.savez_compressed(directory/'failure_state.npz',t=0.,expected_time=.001,
                                qpos=model.qpos0,qvel=np.zeros(model.nv))
            audit = ComparisonAudit(directory)
            audit_prefix(audit,arrays,summary,directory,model)
            self.assertFalse(audit.report['failures'],msg=audit.report['checks'])
            summary['failed_attempt_time_s'] = .0011
            audit = ComparisonAudit(directory)
            audit_prefix(audit,arrays,summary,directory,model)
            self.assertIn('completion_or_explicit_retained_termination_metadata',audit.report['failures'])

    def test_refinement_theta_per_parameter_and_vector_pose_difference(self):
        import json,tempfile
        coarse = np.array([0.,1.,2.]);fine = np.arange(0.,2.01,.5)
        drift = np.arange(1.,17.).reshape(2,8)*.01
        shift = np.arange(16.).reshape(2,8)*.001
        delta_pose = np.array([.0001,.0002,-.0002,.01,.02,.02])
        reference_pose = np.array([.003,.004,0.,.1,0.,0.])
        reference = dict(t=coarse,error=np.broadcast_to(reference_pose,(len(coarse),6)),
            internal_force=np.zeros((len(coarse),2,6)),desired_internal_force=np.zeros((len(coarse),2,6)),
            torque_applied=np.zeros((len(coarse),12)),theta=.9+coarse[:,None,None]*drift)
        candidate = dict(t=fine,error=np.broadcast_to(reference_pose+delta_pose,(len(fine),6)),
            internal_force=np.zeros((len(fine),2,6)),desired_internal_force=np.zeros((len(fine),2,6)),
            torque_applied=np.zeros((len(fine),12)),theta=.9+fine[:,None,None]*drift+shift)
        with tempfile.TemporaryDirectory() as task_directory:
            directory = Path(task_directory)
            np.savez_compressed(directory/'trajectory.npz',**reference)
            (directory/'summary.json').write_text(json.dumps(dict(method='proposed')))
            result = reference_sensitivity(candidate,dict(method='proposed',has_adaptation=True),directory)
            theta = result['theta_sensitivity']
            np.testing.assert_allclose(theta['rms_parameter_trace_difference'],np.abs(shift),atol=1e-15)
            np.testing.assert_allclose(theta['maximum_absolute_parameter_trace_difference'],np.abs(shift),atol=1e-15)
            np.testing.assert_allclose(theta['final_common_parameter_difference_l2_by_arm'],np.linalg.norm(shift,axis=1))
            self.assertAlmostEqual(result['position_orientation_trace_difference']['position_vector_rms_m'],.0003)
            self.assertAlmostEqual(result['position_orientation_trace_difference']['orientation_coordinate_vector_rms_rad'],.03)
            self.assertNotAlmostEqual(.0003,np.linalg.norm(reference_pose[:3]+delta_pose[:3])-np.linalg.norm(reference_pose[:3]))
            (directory/'summary.json').write_text(json.dumps(dict(method='external_ppc')))
            candidate['theta'] = np.full_like(candidate['theta'],np.nan)
            result = reference_sensitivity(candidate,dict(method='external_ppc',has_adaptation=False),directory)
            self.assertFalse(result['theta_sensitivity']['applicable'])


if __name__ == '__main__':
    unittest.main()
