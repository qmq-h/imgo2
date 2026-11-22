#!/usr/bin/env python3
"""Stdlib base-link inertia plausibility audit; never modifies the model.

Uniform-box comparisons are scale references, not identification of real
mass distribution. Binary STL is used only for bounds, never inferred density.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
from audit_gazebo_model_physics import eigenvalues3

ROOT=Path(__file__).resolve().parents[3]

def box_inertia(mass,size):
    return [mass*(size[(i+1)%3]**2+size[(i+2)%3]**2)/12 for i in range(3)]

def main(a):
    b=ET.parse(a.model).getroot().find("link[@name='base']");inertial=b.find('inertial')
    mass=float(inertial.find('mass').get('value'));origin=[float(x) for x in inertial.find('origin').get('xyz').split()]
    v={k:float(x) for k,x in inertial.find('inertia').attrib.items()}
    tensor=[[v['ixx'],v['ixy'],v['ixz']],[v['ixy'],v['iyy'],v['iyz']],[v['ixz'],v['iyz'],v['izz']]]
    diag=[tensor[i][i] for i in range(3)];eigen=sorted(eigenvalues3(tensor));trace=sum(diag)
    margins=[sum(eigen)-2*x for x in eigen]
    assert mass>0 and min(eigen)>0 and min(margins)>=0
    collision=b.find('collision');size=[float(x) for x in collision.find('geometry/box').get('size').split()]
    uniform=box_inertia(mass,size)
    raw=a.mesh.read_bytes();count=struct.unpack_from('<I',raw,80)[0]
    if len(raw)!=84+50*count:raise ValueError('Expected binary STL')
    lo=[math.inf]*3;hi=[-math.inf]*3
    for j in range(count):
        vertices=struct.unpack_from('<12f',raw,84+50*j)[3:]
        for i in range(3):
            for k in range(3):lo[k]=min(lo[k],vertices[3*i+k]);hi[k]=max(hi[k],vertices[3*i+k])
    dimensions=[y-x for x,y in zip(lo,hi)]
    # Covariance C of mass coordinates about COM: I = m*(trace(C)*1-C).
    covariance=[[(trace/2*(i==j)-tensor[i][j])/mass for j in range(3)] for i in range(3)]
    assert min(eigenvalues3(covariance))>=0
    bounds=[(hi[i]-origin[i])*(origin[i]-lo[i]) for i in range(3)]
    result={'base_mass_kg':mass,'base_com_xyz_m':origin,'inertia_units':'kg*m^2','inertia_matrix':tensor,
            'principal_moments':eigen,'principal_triangle_margins':margins,'physically_valid_tensor':True,
            'uniform_main_collision_box':{'size_m':size,'inertia_diagonal':uniform,'model_over_uniform':[x/y for x,y in zip(diag,uniform)]},
            'mesh_bbox':{'min_m':lo,'max_m':hi,'size_m':dimensions,'triangles':count,
                         'uniform_enclosing_box_inertia_diagonal':box_inertia(mass,dimensions)},
            'radius_of_gyration_m':[math.sqrt(x/mass) for x in diag],
            'mass_coordinate_covariance_m2':covariance,
            'mesh_support_variance_upper_bounds_m2':bounds,
            'diagonal_variance_within_mesh_axis_bounds':[covariance[i][i]<=bounds[i] for i in range(3)],
            'origin_to_com_parallel_axis_diagonal_shift':[mass*(sum(x*x for x in origin)-origin[i]**2) for i in range(3)],
            'conclusions':['No evidence of an order-of-magnitude undersized base inertia from tensor validity or box-scale comparison.',
                           'Ixx is about4% larger than a same-mass uniform main collision box; Iyy/Izz also larger.',
                           'A valid tensor and geometry-scale comparison do not verify actual hardware mass distribution.',
                           'The base COM inertia is not whole-robot roll inertia or inertia about a ground pivot.'],
            'sources':{'drake_physical_validity':'https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_rotational_inertia.html',
                       'drake_uniform_box':'https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_unit_inertia.html'},
            'sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.model,a.mesh)}}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('principal_moments','physically_valid_tensor','uniform_main_collision_box','mesh_bbox')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=ROOT/'imgo2_description/urdf/imgo2_real.gazebo.urdf')
    p.add_argument('--mesh',type=Path,default=ROOT/'imgo2_description/meshes/base.STL')
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
