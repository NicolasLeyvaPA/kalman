import { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation,
} from 'd3-force';
import { select } from 'd3-selection';
import { drag } from 'd3-drag';
import { zoom } from 'd3-zoom';
import { scoreColor } from '../utils/colors';

const MIN_RADIUS = 4;
const MAX_RADIUS = 20;

function nodeRadius(node) {
  const v = Math.sqrt(Number(node.total_volume) || 1) / 60;
  return Math.max(MIN_RADIUS, Math.min(MAX_RADIUS, v));
}

export default function BubbleGraph({ nodes, edges, minHeight = 500 }) {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const simRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!nodes || !edges || !containerRef.current) return undefined;

    const ro = new ResizeObserver(() => render());
    ro.observe(containerRef.current);
    render();
    return () => {
      ro.disconnect();
      simRef.current?.stop();
    };

    function render() {
      const { width } = containerRef.current.getBoundingClientRect();
      const height = Math.max(minHeight, width * 0.6);

      const svg = select(svgRef.current)
        .attr('width', width)
        .attr('height', height);
      svg.selectAll('*').remove();

      const g = svg.append('g');
      svg.call(
        zoom().scaleExtent([0.2, 5]).on('zoom', (event) => {
          g.attr('transform', event.transform);
        }),
      );

      const nodeData = nodes.map((n) => ({ ...n }));
      const edgeData = edges.map((e) => ({ ...e }));

      const sim = forceSimulation(nodeData)
        .force('link', forceLink(edgeData).id((d) => d.id).distance(80).strength(0.5))
        .force('charge', forceManyBody().strength(-120))
        .force('center', forceCenter(width / 2, height / 2))
        .force('collide', forceCollide().radius((d) => nodeRadius(d) + 4));
      simRef.current = sim;

      const link = g.append('g')
        .selectAll('line')
        .data(edgeData)
        .enter().append('line')
        .attr('stroke', '#333')
        .attr('stroke-opacity', 0.4)
        .attr('stroke-width', (d) => 0.5 + (Number(d.weight) || 0) * 2);

      const node = g.append('g')
        .selectAll('circle')
        .data(nodeData)
        .enter().append('circle')
        .attr('r', nodeRadius)
        .attr('fill', (d) => scoreColor(d.insider_score))
        .attr('stroke', '#000')
        .attr('stroke-width', 1)
        .style('cursor', 'pointer')
        .on('click', (_, d) => navigate(`/wallet/${d.id}`));

      node.append('title')
        .text((d) =>
          `${d.label}\nscore ${Number(d.insider_score).toFixed(2)}\ncluster ${d.cluster_id}`,
        );

      const labels = g.append('g')
        .selectAll('text')
        .data(nodeData.filter((n) => Number(n.insider_score) >= 0.5))
        .enter().append('text')
        .text((d) => d.label)
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
          }),
      );

      sim.on('tick', () => {
        link
          .attr('x1', (d) => d.source.x).attr('y1', (d) => d.source.y)
          .attr('x2', (d) => d.target.x).attr('y2', (d) => d.target.y);
        node.attr('cx', (d) => d.x).attr('cy', (d) => d.y);
        labels.attr('x', (d) => d.x).attr('y', (d) => d.y - 12);
      });
    }
  }, [nodes, edges, minHeight, navigate]);

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef} className="bg-bg-card border border-border rounded w-full block" />
    </div>
  );
}
