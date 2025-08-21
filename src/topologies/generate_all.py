import os, json
from .starlink import generate_starlink_topology
from .internet import generate_internet_topology
from .dsn import generate_dsn_topology

def main():
    outdir = os.path.join('results', 'topologies')
    os.makedirs(outdir, exist_ok=True)

    starlink = generate_starlink_topology(num_nodes=200, num_orbital_planes=8,
                                          nodes_per_plane=25, altitude_km=550)
    internet = generate_internet_topology(num_nodes=100, num_regions=5,
                                          nodes_per_region=20)
    dsn      = generate_dsn_topology(num_nodes=30)

    with open(os.path.join(outdir, 'STARLINK-200.json'), 'w') as f:
        json.dump(starlink, f, indent=2)
    with open(os.path.join(outdir, 'INTERNET-100.json'), 'w') as f:
        json.dump(internet, f, indent=2)
    with open(os.path.join(outdir, 'DSN-30.json'), 'w') as f:
        json.dump(dsn, f, indent=2)

def generate_all_topologies():
    main()

if __name__ == "__main__":
    main()

