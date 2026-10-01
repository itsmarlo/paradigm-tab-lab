export type Row = { values: number[]; label: number };
export const initialRows: Row[] = [{values:[25,42,2,18],label:0},{values:[46,85,8,64],label:1},{values:[33,61,4,35],label:1}];
export const limits = [80,150,20,100];
export const fields = ['Age','Income','Tenure','Transactions'];
export function contextPrediction(rows: Row[], query: number[]) {
 if (!rows.length) return 0.5;
 const weights = rows.map(r => Math.exp(-12*r.values.reduce((s,v,i)=>s+((v-query[i])/limits[i])**2,0)));
 return rows.reduce((s,r,i)=>s+weights[i]*r.label,0)/weights.reduce((s,w)=>s+w,0);
}
export type Algorithm = 'Logistic Regression' | 'Random Forest' | 'Gradient Boosting';
export function trainDemo(rows:Row[],algorithm:Algorithm) {
 const centers = [0,1].map(label=>fields.map((_,i)=> {const group=rows.filter(r=>r.label===label);return group.length?group.reduce((s,r)=>s+r.values[i]/limits[i],0)/group.length:0.5;}));
 return (query:number[]) => {const distances=centers.map(c=>c.reduce((s,v,i)=>s+(v-query[i]/limits[i])**2,0));const signal=distances[0]-distances[1];const scale=algorithm==='Logistic Regression'?8:algorithm==='Random Forest'?12:16;return 1/(1+Math.exp(-scale*signal));};
}
export function tokenChoices(text:string) { const last=text.trim().split(/\s+/).at(-1)?.toLowerCase();return last==='the'?[['world',.46],['model',.32],['data',.22]] as const:last==='learn'?[['from',.68],['patterns',.21],['together',.11]] as const:[['learn',.48],['connect',.31],['evolve',.21]] as const; }
