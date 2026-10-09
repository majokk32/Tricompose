#!/usr/bin/env python
"""Preserve native PLY grammar namespace through the disk-output wrapper.

The v1 attempt is unchanged. PLY normally inspects its direct caller; wrapping
yacc displaced that frame. Explicitly select the same official parser module,
without changing any grammar/rule, report target or parsing-model version.
"""
from __future__ import print_function
import importlib.util
import json
from pathlib import Path
import sys


def load_v1():
    path = Path(__file__).with_name('initialize_chexpert_negbio_legacy_v1.py')
    spec = importlib.util.spec_from_file_location('_legacy_initialization_v1', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    legacy = load_v1()
    legacy.guard()
    import ply.yacc
    original_yacc = ply.yacc.yacc
    def same_official_grammar(*args, **kwargs):
        native_module = sys.modules.get('negbio.ngrex.parser')
        if native_module is None:
            raise ValueError('official_ngrex_grammar_module_required')
        if 'module' in kwargs and kwargs['module'] is not native_module:
            raise ValueError('official_ngrex_grammar_module_required')
        kwargs['module'] = native_module
        return original_yacc(*args, **kwargs)
    ply.yacc.yacc = same_official_grammar
    original_initialize = legacy.initialize
    def initialize_v2():
        result = original_initialize()
        result['schema_version'] = 'tricompose-chexpert-negbio-offline-initialization-v2'
        result['legacy_worker_sha256'] = result['worker_sha256']
        result['worker_sha256'] = legacy.sha(Path(__file__))
        result['native_ply_grammar_namespace_explicit'] = 'negbio.ngrex.parser'
        result['ply_grammar_or_rules_changed'] = False
        return result
    legacy.initialize = initialize_v2
    legacy.main()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'offline_initialization_failed', 'error_type': type(error).__name__,
                          'source_text_exposed': False}), file=sys.stderr)
        raise SystemExit(2)
