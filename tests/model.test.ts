import { test } from 'node:test';
import assert from 'node:assert/strict';
import { contextPrediction, initialRows, trainDemo } from '../src/model.ts';
test('context labels affect prediction without training',()=>{const query=initialRows[0].values;assert.ok(contextPrediction(initialRows,query)<.5);assert.ok(contextPrediction(initialRows.map(r=>({...r,label:1})),query)>.99);});
test('empty context has a neutral score',()=>assert.equal(contextPrediction([], [30,50,4,20]),.5));
test('trained snapshot is unaffected by subsequent dataset edits',()=>{const rows=structuredClone(initialRows);const model=trainDemo(rows,'Logistic Regression');const before=model([35,65,4,40]);rows[0].label=1;rows[0].values=[80,150,20,100];assert.equal(model([35,65,4,40]),before);});
test('prediction remains finite and bounded at feature limits',()=>{for(const query of [[0,0,0,0],[80,150,20,100]]){const p=contextPrediction(initialRows,query);assert.ok(Number.isFinite(p)&&p>=0&&p<=1);}});
