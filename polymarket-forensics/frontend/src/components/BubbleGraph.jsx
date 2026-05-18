import { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  forceSimulation, forceManyBody, forceCenter, forceLink, forceCollide,
} from 'd3-force';
import { select } from 'd3-selection';
import { drag } from 'd3-drag';
import { zoom } from 'd3-zoom';
import { scoreColor } from '../utils/colors';

export default function BubbleGraph({ nodes, edges, width = 1000, height = 700 }) {
  const ref = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!nodes || !edges) return;
    const svg = select(ref.current);
    svg.selectAll('*').remove();

    const g = svg.append('g');

    svg.call(
      zoom().scaleExtent([0.2, 5]).on('zoom', (event) => {
        g.attr('transform', event.transform);
      })
    );

    const sim = forceSimulation(nodes)
      .force('link', forceLink(edges).id(d => d.id).distance(80).strength(0.5))
      .force('charge', forceManyBody().strength(-120))
      .force('center', forceCenter(width / 2, height / 2))
      .force('collide', forceCollide().radius(d => Math.sqrt(d.total_volume || 1) / 50 + 8));

    const link = g.append('g')
      .selectAll('line')
      .data(edges)
      .enter().append('line')
      .attr('stroke', '#333')
      .attr('stroke-opacity', 0.4)
      .attr('stroke-width', d => 0.5 + (d.weight || 0) * 2);

    const node = g.append('g')
      .selectAll('circle')
      .data(nodes)
      .enter().append('circle')
      .attr('r', d => {
        const v = Math.sqrt(d.total_volume || 1) / 60;
        return Math.max(4, Math.min(20, v));
      })
      .attr('fill', d => scoreColor(d.insider_score))
      .attr('stroke', '#000')
      .attr('stroke-width', 1)
      .style('cursor', 'pointer')
      .on('click', (_, d) => navigate(`/wallet/${d.id}`));

    node.append('title')
      .text(d => `${d.label}\nscore ${d.insider_score?.toFixed(2)}\ncluster ${d.cluster_id}`);

    const labels = g.append('g')
      .selectAll('text')
      .data(nodes.filter(n => (n.insider_score || 0) >= 0.5))
      .enter().append('text')
      .text(d => d.label)
      .attr('font-size', 9)
      .attr('fill', '#888')
      .attr('text-anchor', 'middle')
      .attr('pointer-events', 'none');

    node.call(
      drag()
        .on('start', (event, d) => {
          if (!event.active) sim.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on('end', (event, d) => {
          if (!event.active) sim.alphaTarget(0);
          d.fx = null; d.fy = null;
        })
    );

    sim.on('tick', () => {
      link
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('cx', d => d.x).attr('cy', d => d.y);
      labels.attr('x', d => d.x).attr('y', d => d.y - 12);
    });

    return () => sim.stop();
  }, [nodes, edges, width, height, navigate]);

  return (
    <svg ref={ref} width={width} height={height}
         className="bg-bg-card border border-border rounded" />
  );
}
