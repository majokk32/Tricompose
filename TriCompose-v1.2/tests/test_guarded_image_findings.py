"""Invented images/metadata and mocked callbacks; no models or patient inputs."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prospective_guard_fixture',ROOT/'tools/verify_guarded_image_findings.py')
w=importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
g=w.guard
SHA='a'*64


class Image:
    mode='RGB'
    width=height=64
    def __init__(self, uniform=False, value=0):
        self.uniform,self.value=uniform,value
    def tobytes(self):
        if self.uniform:return bytes([self.value])*(64*64*3)
        return bytes([self.value,self.value+1])*(64*64*3//2)


def guarded(image,sha=SHA):
    extrema=((image.value,image.value if image.uniform else image.value+1),)*3
    current=g.metadata_guard(sha,width=64,height=64,mode='RGB',format_name='PNG',frames=1,extrema=extrema)
    return g.receipt(sha,current['status'],current['reason'],evidence=current['native_image_metadata'],
        pixel_sha256=g.normalized_pixel_sha(image))


def raw(failed=False,state='negative'):
    return {'contract_status':'failed_unavailable' if failed else 'complete',
        'states':None if failed else dict.fromkeys(g.HEADS,state),'failure_reason':'invalid' if failed else None,
        'input_tokens':60,'output_tokens':80,'token_limit_reached':False,'response_sha256':'b'*64,
        'elapsed_seconds':0.1,'independent_clinical_validation':False}


def plan_for(images,budget=10):
    items=[{'observation_id':g.normalized_pixel_sha(image),'path':'/invented/'+str(index),'sha256':SHA,
        'width':64,'height':64} for index,image in enumerate(images)]
    return {'inputs':sorted(items,key=lambda r:r['observation_id']),'max_model_calls':budget,
        'image_prompt_sha256':'c'*64,'logical_slots':[{'observation_id':r['observation_id'],'arm':'original'}
            for r,image in zip(items,images) if not image.uniform]}


def hook_for(images):
    paths={'/invented/'+str(i):image for i,image in enumerate(images)}
    def hook(path,sha,callback,**kwargs):
        image=paths[path]
        current=guarded(image,sha)
        called=current['basic_comparison_permitted']
        return {'guard':current,'callback_invoked':called,'callback_result':callback(image) if called else None}
    return hook


def inventory_fixture():
    originals={'image_inputs':[{'cxr_candidate_id':'image_'+str(i),'cxr_sha256':str(i)*64,
        'path':'/invented/source_'+str(i),'image_file_stats':[12,34]} for i in range(6)]}
    logical=w.controls.logical_slots(originals['image_inputs'])
    slots=[]
    for slot in logical:
        index=int(slot['cxr_candidate_id'].split('_')[-1])
        size=64 if index%2==0 else 128
        key=(str(index)*64 if slot['arm']=='original' else
            ('a' if slot['arm']=='uniform_black' else 'b')+str(size))
        slots.append({**slot,'observation_id':key,'width':size,'height':size})
    controls=[{'observation_id':letter+str(size),'path':letter+str(size)+'.png',
        'sha256':letter*64,'width':size,'height':size} for letter in ('a','b') for size in (64,128)]
    manifest={'artifacts':{r['path']:{'sha256':r['sha256']} for r in controls}}
    return originals,{'slots':slots,'saved_uniform_controls':controls},manifest


class GuardedVerifierTests(unittest.TestCase):
    def invoke(self,images,cb,budget=10):
        plan=plan_for(images,budget)
        with patch.object(g,'guarded_invoke',side_effect=hook_for(images)):
            return w.invoke_inputs(plan,cb,Image=None),plan

    def test_uniform_inputs_never_reach_model(self):
        images=[Image(True,0),Image(True,255),Image()]
        cb=Mock(return_value=raw())
        (records,calls),_=self.invoke(images,cb)
        self.assertEqual(calls,1)
        cb.assert_called_once_with(images[2])
        blocked=[r for r in records if not r['model_called']]
        self.assertEqual(len(blocked),2)
        for r in blocked:
            self.assertEqual(r['contract_status'],'blocked_before_model_call')
            for k in ('states','response_sha256','input_tokens','output_tokens','token_limit_reached'):
                self.assertIsNone(r[k])

    def test_all_invalid_inputs_need_no_callback_and_no_model_loading(self):
        cb=Mock(side_effect=AssertionError('must_not_call'))
        (records,calls),_=self.invoke([Image(True,0),Image(True,255)],cb)
        cb.assert_not_called()
        self.assertEqual(calls,0)
        self.assertEqual(len(records),2)

    def test_callback_receives_exact_pixels_only(self):
        image=Image()
        cb=Mock(return_value=raw())
        (records,calls),_=self.invoke([image],cb)
        cb.assert_called_once_with(image)
        self.assertEqual(calls,1)
        self.assertEqual(records[0]['states'],raw()['states'])
        self.assertFalse(records[0]['independent_clinical_validation'])

    def test_unknown_uncertain_and_failed_callbacks_remain_distinct(self):
        for value in (raw(state='unknown'),raw(state='uncertain'),raw(failed=True)):
            (records,calls),_=self.invoke([Image()],Mock(return_value=value))
            self.assertEqual(records[0]['states'],value['states'])
            self.assertEqual(records[0]['contract_status'],value['contract_status'])
            self.assertEqual(calls,1)

    def test_call_budget_applies_before_model_invocation(self):
        cb=Mock(return_value=raw())
        with self.assertRaisesRegex(ValueError,'budget'):
            self.invoke([Image(value=0),Image(value=2)],cb,budget=1)
        self.assertEqual(cb.call_count,1)

    def test_zero_call_budget_still_allows_all_blocked_records(self):
        cb=Mock()
        (records,calls),_=self.invoke([Image(True)],cb,budget=0)
        self.assertEqual(calls,0)
        self.assertEqual(len(records),1)
        cb.assert_not_called()

    def test_duplicate_or_unsorted_inputs_refused(self):
        for images in ([Image(),Image()], [Image(),Image(value=2)]):
            plan=plan_for(images)
            if len(set(r['observation_id'] for r in plan['inputs']))==len(images):
                plan['inputs'].reverse()
            with patch.object(g,'guarded_invoke') as hook:
                with self.assertRaises(ValueError):w.invoke_inputs(plan,Mock(),Image=None)
                hook.assert_not_called()

    def test_wrong_pixel_identity_or_dimensions_refused_before_callback(self):
        for key in ('observation_id','width'):
            image=Image()
            plan=plan_for([image])
            plan['inputs'][0][key]='d'*64 if key=='observation_id' else 65
            cb=Mock(return_value=raw())
            with patch.object(g,'guarded_invoke',side_effect=hook_for([image])):
                with self.assertRaises(ValueError):w.invoke_inputs(plan,cb,Image=None)
            cb.assert_not_called()

    def test_unsanitized_or_clinical_promoted_callback_refused(self):
        for value in ({**raw(),'generated_text':'invented'}, {**raw(),'independent_clinical_validation':True},
                {**raw(),'states':dict.fromkeys(g.HEADS,'absent')}):
            with self.assertRaises(ValueError):self.invoke([Image()],Mock(return_value=value))

    def test_model_exception_is_not_retried_or_successful_unknown(self):
        cb=Mock(side_effect=RuntimeError('invented_backend_failure'))
        with self.assertRaises(RuntimeError):self.invoke([Image()],cb)
        self.assertEqual(cb.call_count,1)

    def test_input_plan_unchanged_by_execution(self):
        images=[Image(),Image(True)]
        plan=plan_for(images)
        before=copy.deepcopy(plan)
        with patch.object(g,'guarded_invoke',side_effect=hook_for(images)):
            w.invoke_inputs(plan,Mock(return_value=raw()),Image=None)
        self.assertEqual(plan,before)

    def test_metadata_inventory_deduplicates_without_opening_pixels(self):
        original,realized,manifest=inventory_fixture()
        before=copy.deepcopy((original,realized,manifest))
        with patch.object(w,'require_inside',side_effect=lambda p,*a,**k:Path(p)), \
                patch.object(Path,'stat',return_value=SimpleNamespace(st_size=12,st_mtime_ns=34)), \
                patch.object(Path,'is_file',return_value=True), patch.object(Path,'read_bytes') as pixels:
            result=w.inventory(original,realized,manifest)
        self.assertEqual(len(result),10)
        self.assertEqual([r['observation_id'] for r in result],sorted(r['observation_id'] for r in result))
        self.assertEqual((original,realized,manifest),before)
        pixels.assert_not_called()

    def test_changed_arm_order_or_control_hash_refused(self):
        for reason in ('order','hash','size'):
            original,realized,manifest=inventory_fixture()
            if reason=='order':realized['slots'].reverse()
            elif reason=='hash':manifest['artifacts'][realized['saved_uniform_controls'][0]['path']]['sha256']='x'*64
            else:realized['saved_uniform_controls'][0]['width']=65
            with patch.object(w,'require_inside',side_effect=lambda p,*a,**k:Path(p)), \
                    patch.object(Path,'stat',return_value=SimpleNamespace(st_size=12,st_mtime_ns=34)), \
                    patch.object(Path,'is_file',return_value=True):
                with self.assertRaises(ValueError):w.inventory(original,realized,manifest)

    def test_preparation_uses_metadata_not_pixels_or_prediction_states(self):
        original,realized,manifest=inventory_fixture()
        for key in ('model_path','model_file_stats','model_file_sha256','finding_order','prompt_version',
                'image_prompt_sha256','min_pixels','max_pixels','max_new_tokens','model_retries','seed',
                'do_sample','min_vram_gib'):
            original[key]=key
        manifest['new_model_calls']=10
        manifest['artifacts']['predictions.json']={'sha256':SHA}
        gm={'new_model_calls':0,'source_sha256':{'guard':SHA,'worker':SHA},
            'decoder_source_paths':{},'decoder_source_sha256':{}}
        with patch.object(w.inputs,'ORIGINAL_PLAN_SHA',SHA),patch.object(w,'sha256_file',return_value=SHA), \
                patch.object(w.cached,'load_plan',return_value=(original,{})), \
                patch.object(w.inputs,'fixed_parent',side_effect=[(manifest,{'realized_inputs.json':realized},{}),(gm,{}, {})]), \
                patch.object(w,'require_inside',side_effect=lambda p,*a,**k:Path(p)), \
                patch.object(Path,'stat',return_value=SimpleNamespace(st_size=12,st_mtime_ns=34)), \
                patch.object(Path,'is_file',return_value=True),patch.object(Path,'read_bytes') as pixels, \
                patch.object(w.cached,'bounded_json') as states:
            plan,sources=w.prepare()
        self.assertEqual(len(plan['inputs']),10)
        self.assertEqual(plan['max_model_calls'],10)
        self.assertFalse(plan['old_prediction_states_parsed_in_prepare'])
        self.assertIn('baseline_prediction_bytes',sources)
        pixels.assert_not_called()
        states.assert_not_called()

    def test_unsealed_plan_or_changed_program_refused_before_rebuild(self):
        for reason in ('schema','artifact','program'):
            manifest={'schema_version':w.SCHEMA+'-plan','artifacts':{'plan.json':{'sha256':SHA}},
                'source_paths':{'fixture':'/invented/program'},'source_sha256':{'fixture':SHA}}
            if reason=='schema':manifest['schema_version']='wrong'
            with patch.object(w,'require_inside',side_effect=lambda p,*a,**k:Path(p)), \
                    patch.object(w.cached,'bounded_json',side_effect=[manifest,{}]), \
                    patch.object(w,'sha256_file',side_effect=lambda p: ('x'*64 if
                        (reason=='artifact' or (reason=='program' and p.name=='program')) else SHA)), \
                    patch.object(w,'prepare') as prepare:
                with self.assertRaises(ValueError):w.load_plan(Path('/invented/run'))
                prepare.assert_not_called()

    def test_guard_only_result_not_clinical_efficiency(self):
        images=[Image(),Image(True)]
        (records,calls),plan=self.invoke(images,Mock(return_value=raw()))
        baseline={'records':[{'observation_id':g.normalized_pixel_sha(im),**raw()} for im in images],
            'frozen':True,'image_only':True,'image_prompt_sha256':plan['image_prompt_sha256'],
            'model_received_arm_names_ehr_reports_ids_scores_or_expected_answers':False}
        summary,rows=w.analyze(records,plan,baseline,calls)
        self.assertEqual(summary['same_input_call_count_difference'],1)
        self.assertEqual(summary['original_readout_matching_states'],8)
        self.assertEqual(summary['blocked_before_model_call'],1)
        self.assertIsNone(summary['clinical_compute_savings'])
        for key in ('clinical_acceptance','primary_metric_eligible','selection_changed','regeneration_authorized'):
            self.assertFalse(summary[key])
        self.assertEqual(len(rows),1)

    def test_predictions_fsynced_before_old_states_parsed(self):
        events=[]
        file=Mock()
        file.open.return_value.__enter__=Mock(return_value=SimpleNamespace(fileno=lambda:17))
        file.open.return_value.__exit__=Mock(return_value=None)
        def baseline(*args):
            self.assertIn('fsync',events)
            events.append('baseline')
            return {}
        with patch.object(w,'write_private_json',return_value=file), \
                patch.object(w.os,'fsync',side_effect=lambda fd:events.append('fsync')), \
                patch.object(w,'sha256_file',return_value=SHA), \
                patch.object(w.cached,'bounded_json',side_effect=baseline), \
                patch.object(w,'analyze',return_value=({},[])):
            w.freeze_then_compare(Path('/invented'),[],{'image_prompt_sha256':SHA},0)
        self.assertEqual(events,['fsync','baseline'])

    def test_fsync_failure_never_reads_baseline(self):
        file=Mock()
        file.open.return_value.__enter__=Mock(return_value=SimpleNamespace(fileno=lambda:17))
        file.open.return_value.__exit__=Mock(return_value=None)
        with patch.object(w,'write_private_json',return_value=file), \
                patch.object(w.os,'fsync',side_effect=OSError),patch.object(w.cached,'bounded_json') as baseline:
            with self.assertRaises(OSError):
                w.freeze_then_compare(Path('/invented'),[],{'image_prompt_sha256':SHA},0)
            baseline.assert_not_called()

    def test_invalid_baseline_cannot_be_called_repeatability(self):
        image=Image()
        (records,calls),plan=self.invoke([image],Mock(return_value=raw()))
        for reason in ('missing','bad_state','cap','prompt','context'):
            baseline={'records':[{'observation_id':g.normalized_pixel_sha(image),**raw()}],
                'frozen':True,'image_only':True,'image_prompt_sha256':plan['image_prompt_sha256'],
                'model_received_arm_names_ehr_reports_ids_scores_or_expected_answers':False}
            if reason=='missing':baseline['records'][0]['states'].pop('edema')
            elif reason=='bad_state':baseline['records'][0]['states']['edema']='absent'
            elif reason=='cap':baseline['records'][0]['token_limit_reached']=True
            elif reason=='prompt':baseline['image_prompt_sha256']='x'*64
            else:baseline['model_received_arm_names_ehr_reports_ids_scores_or_expected_answers']=True
            with self.assertRaises(ValueError):w.analyze(records,plan,baseline,calls)

    def test_approval_guard_before_allocation_or_metadata(self):
        args=SimpleNamespace(mode='run',allow_guarded_image_verification=False,output_root=Path('/invented'),run_id='test')
        with patch.object(w.cached,'require_slurm',side_effect=RuntimeError), \
                patch.object(w,'new_atomic_run') as new:
            with self.assertRaises(RuntimeError):w.execute(args)
            new.assert_not_called()

    def test_overwrite_refused_before_metadata(self):
        args=SimpleNamespace(mode='prepare',allow_guarded_image_verification=False,output_root=Path('/invented'),run_id='test')
        with patch.object(w.cached,'require_slurm'),patch.object(w,'new_atomic_run',side_effect=FileExistsError), \
                patch.object(w,'prepare') as prepare:
            with self.assertRaises(FileExistsError):w.execute(args)
            prepare.assert_not_called()

    def test_failed_preparation_discards_only_new_temporary(self):
        args=SimpleNamespace(mode='prepare',allow_guarded_image_verification=False,output_root=Path('/invented'),run_id='test')
        with patch.object(w.cached,'require_slurm'), \
                patch.object(w,'new_atomic_run',return_value=(Path('/invented/tmp'),Path('/invented/run'))), \
                patch.object(w,'prepare',side_effect=ValueError),patch.object(w,'discard_atomic_run') as discard:
            with self.assertRaises(ValueError):w.execute(args)
            discard.assert_called_once_with(Path('/invented/tmp'))

    def test_unapproved_gpu_run_fails_before_loading_plan(self):
        with patch.object(w.cached,'require_slurm',side_effect=RuntimeError),patch.object(w,'load_plan') as load:
            with self.assertRaises(RuntimeError):w.run(Path('/invented'),Path('/invented/tmp'),approved=False)
            load.assert_not_called()


if __name__=='__main__':unittest.main()
