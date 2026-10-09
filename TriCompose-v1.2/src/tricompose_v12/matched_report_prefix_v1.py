"""Fixed-image report acquisition control with the unchanged V1 commit guard.

Not a new repair policy or clinical localizer. No unrequested candidates or
secondary scores are exposed. Unlike targeted stopping, the fixed schedule
acquires every affordable report in source order, even if no target remains.
"""
from copy import deepcopy
from . import probe_repair_v1 as p

VERSION = 'tricompose-matched-guard-report-prefix-v1'


class FixedReportPrefix(p.ProbeController):
    def __init__(self, initial, report_order, budget):
        image = (initial['lineage']['cxr_model_id'], initial['lineage']['cxr_seed'])
        super().__init__(initial, [image], report_order, budget, feedback_enabled=False)
        self.fixed_image = image

    def propose(self):
        if self.pending is not None:
            raise RuntimeError('complete_reserved_probe_before_next_proposal')
        available = [(*self.fixed_image, report) for report in self.report_order
            if (*self.fixed_image, report) not in self.attempted]
        if not available or self.spent + p.COST['report_probe'] > self.budget:
            return {'action': 'stop', 'reason': 'fixed_report_prefix_complete_or_budget_exhausted',
                'clinical_acceptance': False}
        request = {'schema_version': p.VERSION + '-request', 'step': len(self.history),
            'case_id': self.current['case_id'], 'action': 'report_probe', 'slot': list(available[0]),
            'parent_candidate_id': self.current['candidate_id'],
            'parent_observation_sha256': p.digest(self.current),
            'reserved_simulated_calls': p.COST['report_probe'],
            'reason': 'fixed_report_order_identical_commit_guard', 'uses_unseen_scores': False,
            'uses_report_votes_as_independent_image_truth': False}
        request['request_sha256'] = p.digest(request)
        self.pending = deepcopy(request)
        self.spent += p.COST['report_probe']
        self.attempted.add(tuple(request['slot']))
        return deepcopy(request)

    def result(self):
        result = super().result()
        result.update(schema_version=VERSION, method='fixed_image_guarded_report_prefix',
            scope='matched_commit_guard_fixed_image_development_control',
            same_commit_guard_version=p.VERSION, image_regeneration_allowed=False,
            acquisition_uses_need_signal=False)
        return result
